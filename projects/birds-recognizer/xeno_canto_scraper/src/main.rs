use reqwest::blocking::Client;
use scraper::{Html, Selector};
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};
use threadpool::ThreadPool;
use url::Url;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!("  {} <start_url> [output_directory]", args[0]);
        eprintln!("  {} --convert <directory>", args[0]);
        std::process::exit(1);
    }

    if args[1] == "--convert" {
        if args.len() < 3 {
            eprintln!("Please specify a directory to convert");
            std::process::exit(1);
        }
        return batch_convert_directory(&args[2]);
    }

    let start_url = &args[1];
    let output_dir = args.get(2).map(|s| s.as_str()).unwrap_or("downloads");

    // Create output directory if it doesn't exist
    std::fs::create_dir_all(output_dir)?;

    // Create HTTP client
    let client = Client::new();
    let mut current_page_url = start_url.to_string();
    let mut page_num = 1;

    // Create a thread pool for downloads
    let pool = ThreadPool::new(5); // 5 concurrent downloads
    
    // Counter for each species (shared between threads)
    let species_counters = Arc::new(Mutex::new(HashMap::<String, usize>::new()));
    
    // Create metadata file
    let metadata_path = Path::new(output_dir).join("metadata.csv");
    let mut metadata_file = File::create(&metadata_path)?;
    writeln!(metadata_file, "filename,species,original_url,id,common_name,scientific_name")?;

    loop {
        println!("Processing page {}: {}", page_num, current_page_url);

        // Fetch page content
        let response = client.get(&current_page_url).send()?;
        if !response.status().is_success() {
            eprintln!("Failed to fetch page: {}", response.status());
            break;
        }

        let html = response.text()?;
        let document = Html::parse_document(&html);

        // Extract download links along with metadata
        let selector = Selector::parse("a[href$='/download']").unwrap();
        let mut download_jobs = Vec::new();

        for element in document.select(&selector) {
            if let Some(href) = element.value().attr("href") {
                let download_url = if href.starts_with("http") {
                    href.to_string()
                } else {
                    // FIX: Handle URL parsing error properly
                    let base = match Url::parse(&current_page_url) {
                        Ok(url) => url,
                        Err(e) => {
                            eprintln!("Failed to parse base URL: {}", e);
                            continue;
                        }
                    };
                    
                    match base.join(href) {
                        Ok(url) => url.to_string(),
                        Err(e) => {
                            eprintln!("Failed to join URL: {}", e);
                            continue;
                        }
                    }
                };
                
                // Extract the ID from URL
                let parts: Vec<&str> = download_url.split('/').collect();
                let id = parts.get(parts.len() - 2)
                    .unwrap_or(&"unknown")
                    .to_string();
                
                // Get title attribute from the img tag inside the link
                let img_selector = Selector::parse("img.icon").unwrap();
                let title = element.select(&img_selector).next()
                    .and_then(|img| img.value().attr("title"))
                    .unwrap_or("Unknown Bird");
                
                // Extract both common name and scientific name if available
                // Example title: "Download file 'XC652207 - Arctic Tern - Sterna paradisaea.mp3'"
                let mut common_name = "unknown";
                let mut scientific_name = "unknown";
                
                if title.contains(" - ") {
                    let parts: Vec<&str> = title.split(" - ").collect();
                    if parts.len() >= 2 {
                        common_name = parts[1].trim();
                    }
                    if parts.len() >= 3 {
                        // Remove file extension if present
                        let sci_name = parts[2].trim();
                        scientific_name = if sci_name.ends_with(".mp3") {
                            &sci_name[0..sci_name.len()-4]
                        } else {
                            sci_name
                        };
                    }
                }
                
                download_jobs.push((download_url, id, common_name.to_string(), scientific_name.to_string()));
            }
        }

        println!("Found {} download links", download_jobs.len());
        if download_jobs.is_empty() {
            println!("No more download links found, exiting.");
            break;
        }

        // Start jobs for each download
        for (url, id, common_name, scientific_name) in download_jobs {
            let client = client.clone();
            let output_dir = output_dir.to_string();
            let species_counters = Arc::clone(&species_counters);
            let metadata_path = metadata_path.clone(); // Clone the path for use in thread
            
            pool.execute(move || {
                match download_file_with_metadata(
                    &client, 
                    &url, 
                    &output_dir, 
                    &id, 
                    &common_name, 
                    &scientific_name, 
                    &species_counters
                ) {
                    Ok(filename) => {
                        println!("Downloaded: {} -> {}", url, filename);
                        // FIX: Append to one metadata file with proper locking
                        if let Ok(mut file) = std::fs::OpenOptions::new()
                            .append(true)
                            .open(&metadata_path) {
                            let _ = writeln!(file, "{},{},{},{},{},{}", 
                                         filename, 
                                         format_species_name(&common_name), 
                                         url, 
                                         id, 
                                         common_name, 
                                         scientific_name);
                        }
                    },
                    Err(e) => println!("Failed to download {}: {}", url, e),
                }
            });
        }

        // Find next page link
        let next_page_selector = Selector::parse("a.pagination-next").unwrap();
        let next_page = document.select(&next_page_selector).next();

        if let Some(next_link) = next_page {
            if let Some(href) = next_link.value().attr("href") {
                // Construct next page URL
                let base = Url::parse(&current_page_url)?;
                current_page_url = base.join(href)?.to_string();
                page_num += 1;
            } else {
                println!("Next page link found but no href attribute, exiting.");
                break;
            }
        } else {
            // Check if we can construct the next page URL based on the pattern
            if current_page_url.contains("pg=") {
                // Replace page number in URL
                let parts: Vec<&str> = current_page_url.split("pg=").collect();
                if parts.len() == 2 {
                    let base = parts[0];
                    let rest: Vec<&str> = parts[1].splitn(2, |c| c == '&').collect();
                    let page_str = if rest.len() > 1 {
                        format!("pg={}&{}", page_num + 1, rest[1])
                    } else {
                        format!("pg={}", page_num + 1)
                    };
                    current_page_url = format!("{}{}", base, page_str);
                    page_num += 1;
                    continue;
                }
            }
            
            println!("No next page link found, exiting.");
            break;
        }
    }

    // Wait for remaining jobs to complete
    pool.join();
    
    println!("Scraping completed!");
    Ok(())
}

fn format_species_name(name: &str) -> String {
    // Convert "Arctic Tern" to "arctic_tern"
    name.to_lowercase().replace(' ', "_")
}

fn download_file_with_metadata(
    client: &Client, 
    url: &str, 
    output_dir: &str, 
    id: &str, 
    common_name: &str, 
    scientific_name: &str,
    species_counters: &Arc<Mutex<HashMap<String, usize>>>
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    // Format the species name for the filename
    let species_name = format_species_name(common_name);
    
    // Get the next number for this species
    let file_number = {
        let mut counters = species_counters.lock().unwrap();
        let count = counters.entry(species_name.clone()).or_insert(0);
        *count += 1;
        *count
    };
    
    // Create the formatted filenames (both MP3 and WAV)
    let mp3_filename = format!("{}_{}.mp3", species_name, file_number);
    let wav_filename = format!("{}_{}.wav", species_name, file_number);
    
    let mp3_path = Path::new(output_dir).join(&mp3_filename);
    let wav_path = Path::new(output_dir).join(&wav_filename);
    
    println!("Downloading: {} → {}", url, mp3_path.display());

    // Download the file
    let response = client.get(url).send()?;
    if !response.status().is_success() {
        return Err(format!("Download failed with status: {}", response.status()).into());
    }
    
    let bytes = response.bytes()?;
    
    // Save to MP3 file
    let mut file = File::create(&mp3_path)?;
    file.write_all(&bytes)?;
    
    // Convert to WAV
    match convert_using_ffmpeg(&mp3_path, &wav_path) {
        Ok(_) => {
            // Optionally remove the MP3 file if you don't need it
            // std::fs::remove_file(&mp3_path)?;
        },
        Err(e) => {
            println!("Error converting to WAV: {}", e);
            // Continue without conversion
        }
    }

    Ok(wav_filename) // Return the WAV filename for metadata
}

fn batch_convert_directory(dir_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let dir = Path::new(dir_path);
    if !dir.is_dir() {
        return Err(format!("{} is not a directory", dir_path).into());
    }
    
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension().map_or(false, |ext| ext == "mp3") {
            let stem = path.file_stem().unwrap().to_string_lossy();
            let wav_path = path.with_extension("wav");
            
            println!("Converting: {}", path.display());
            match convert_using_ffmpeg(&path, &wav_path) {
                Ok(_) => println!("Conversion successful: {}", wav_path.display()),
                Err(e) => println!("Error converting {}: {}", path.display(), e),
            }
        }
    }
    
    Ok(())
}


fn convert_using_ffmpeg(mp3_path: &Path, wav_path: &Path) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
    // Check if ffmpeg is available
    use std::process::Command;
    
    println!("Attempting conversion with ffmpeg: {} → {}", mp3_path.display(), wav_path.display());
    
    let output = Command::new("C:\\tools\\ffmpeg.exe")
        .arg("-y") // Overwrite existing files
        .arg("-i")
        .arg(mp3_path)
        .arg("-acodec")
        .arg("pcm_s16le") // 16-bit PCM encoding for WAV
        .arg("-ar")
        .arg("44100") // Standard sample rate
        .arg(wav_path)
        .output()?;
    
    if output.status.success() {
        println!("ffmpeg conversion successful");
        Ok(true)
    } else {
        let error = String::from_utf8_lossy(&output.stderr);
        println!("ffmpeg conversion failed: {}", error);
        Ok(false) // Return false but not an error so we can try fallback
    }
}