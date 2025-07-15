use reqwest::blocking::Client;
use scraper::{Html, Selector};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use threadpool::ThreadPool;
use url::Url;
use csv::{Reader, Writer};

// Struct to hold metadata for a recording
#[derive(Debug, Clone)]
struct RecordingMetadata {
    id: String,
    url: String,
    common_name: String,
    scientific_name: String,
    filename: String,
    species: String, // Normalized species name
    is_downloaded: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!("  {} <start_url> [output_directory] [--delay <ms>]", args[0]);
        eprintln!("  {} --download-only [output_directory] [--delay <ms>]", args[0]);
        eprintln!("  {} --convert <directory>", args[0]);
        std::process::exit(1);
    }

    // Handle conversion command
    if args[1] == "--convert" {
        if args.len() < 3 {
            eprintln!("Please specify a directory to convert");
            std::process::exit(1);
        }
        return batch_convert_directory(&args[2]);
    }

    // Rate limiting settings
    let mut page_delay_ms = 2000; // Default: 2 seconds between page requests
    let mut download_delay_ms = 500; // Default: 0.5 seconds between downloads
    
    // Check for custom delay parameter
    if let Some(delay_index) = args.iter().position(|arg| arg == "--delay") {
        if delay_index + 1 < args.len() {
            if let Ok(delay) = args[delay_index + 1].parse::<u64>() {
                page_delay_ms = delay;
                download_delay_ms = delay / 4;
                println!("Using custom delay: {}ms between pages, {}ms between downloads", 
                         page_delay_ms, download_delay_ms);
            }
        }
    }

    // Check if we're in download-only mode
    let download_only = args[1] == "--download-only";
    
    // Determine output directory based on mode
    let output_dir = if download_only {
        args.get(2)
            .filter(|&arg| !arg.starts_with("--"))
            .map(|s| s.as_str())
            .unwrap_or("downloads")
    } else {
        // Normal mode - first arg is URL
        args.get(2)
            .filter(|&arg| !arg.starts_with("--"))
            .map(|s| s.as_str())
            .unwrap_or("downloads")
    };

    // Create output directory if it doesn't exist
    std::fs::create_dir_all(output_dir)?;
    
    // Create HTTP client
    let client = Client::builder()
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        .build()?;
    
    let metadata_path = Path::new(output_dir).join("metadata.csv");
    
    if download_only {
        // Skip extraction and just load existing metadata
        if !metadata_path.exists() {
            eprintln!("Error: No metadata.csv found in {}. Cannot use --download-only mode.", output_dir);
            std::process::exit(1);
        }
        
        println!("Download-only mode: Using existing metadata from {}", metadata_path.display());
        println!("1. Reading metadata CSV...");
        let metadata = load_existing_metadata(&metadata_path)?;
        
        println!("2. Checking for already downloaded files...");
        let updated_metadata = update_download_status(&metadata, output_dir)?;
        
        println!("3. Saving updated metadata...");
        write_metadata_csv(&metadata_path, &updated_metadata)?;
        
        println!("4. Downloading missing files...");
        download_missing_files(&client, &updated_metadata, output_dir, download_delay_ms)?;
    } else {
        // Normal mode - extract links first
        let start_url = &args[1];
        
        println!("1. Extracting all download links...");
        let download_info = extract_all_download_links(client.clone(), start_url, page_delay_ms)?;
        println!("Found {} total download links", download_info.len());
        
        println!("2. Creating/updating metadata CSV...");
        let metadata = load_or_create_metadata(&metadata_path, &download_info)?;
        
        println!("3. Checking for already downloaded files...");
        let updated_metadata = update_download_status(&metadata, output_dir)?;
        
        println!("4. Saving updated metadata...");
        write_metadata_csv(&metadata_path, &updated_metadata)?;
        
        println!("5. Downloading missing files...");
        download_missing_files(&client, &updated_metadata, output_dir, download_delay_ms)?;
    }
    
    println!("Scraping completed!");
    Ok(())
}

// Extract all download links by crawling pages
fn extract_all_download_links(
    client: Client, 
    start_url: &str, 
    page_delay_ms: u64
) -> Result<Vec<(String, String, String, String)>, Box<dyn std::error::Error>> {
    let mut current_page_url = start_url.to_string();
    let mut page_num = 1;
    let mut download_info = Vec::new();
    
    loop {
        println!("Processing page {}: {}", page_num, current_page_url);

        // Rate limiting: Wait before making the next page request
        thread::sleep(Duration::from_millis(page_delay_ms));

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
        let mut page_downloads = Vec::new();

        for element in document.select(&selector) {
            if let Some(href) = element.value().attr("href") {
                let download_url = if href.starts_with("http") {
                    href.to_string()
                } else {
                    // Handle URL parsing error properly
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
                
                page_downloads.push((download_url, id, common_name.to_string(), scientific_name.to_string()));
            }
        }
        
        let page_downloads_count = page_downloads.len();
        download_info.extend(page_downloads);
        println!("Found {} download links on page {}", page_downloads_count, page_num);
        
        
        if page_downloads_count == 0 {
            println!("No more download links found on page {}, exiting.", page_num);
            break;
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
    
    Ok(download_info)
}

// Load existing metadata or create new metadata with newly found links
fn load_or_create_metadata(
    metadata_path: &Path,
    download_info: &[(String, String, String, String)]
) -> Result<Vec<RecordingMetadata>, Box<dyn std::error::Error>> {
    let mut metadata = Vec::new();
    let mut existing_ids = HashSet::new();
    
    // Read existing metadata if file exists
    if metadata_path.exists() {
        let mut reader = csv::Reader::from_path(metadata_path)?;
        
        for result in reader.records() {
            let record = result?;
            if record.len() >= 6 { // Basic validation
                let filename = record[0].to_string();
                let species = record[1].to_string();
                let url = record[2].to_string();
                let id = record[3].to_string();
                let common_name = record[4].to_string();
                let scientific_name = record[5].to_string();
                // Parse is_downloaded flag if it exists
                let is_downloaded = if record.len() >= 7 {
                    record[6].parse::<bool>().unwrap_or(false)
                } else {
                    false
                };
                
                existing_ids.insert(id.clone());
                
                metadata.push(RecordingMetadata {
                    id,
                    url,
                    common_name,
                    scientific_name,
                    filename,
                    species,
                    is_downloaded,
                });
            }
        }
    }
    
    // Get max number used for each species to avoid duplicates
    let mut species_counters = HashMap::new();
    for meta in &metadata {
        let species = &meta.species;
        if let Some(number_str) = meta.filename.strip_prefix(&format!("{}_", species)) {
            if let Some(number_str) = number_str.strip_suffix(".wav") {
                if let Ok(number) = number_str.parse::<usize>() {
                    let current_max = species_counters.entry(species.clone()).or_insert(0);
                    if number > *current_max {
                        *current_max = number;
                    }
                }
            }
        }
    }
    
    // Add new download links to metadata
    for (url, id, common_name, scientific_name) in download_info {
        // Skip if already exists in metadata
        if existing_ids.contains(id) {
            continue;
        }
        
        // Format the species name
        let species = format_species_name(common_name);
        
        // Generate filename with next available number
        let counter = species_counters.entry(species.clone()).or_insert(0);
        *counter += 1;
        let next_number = *counter;
        
        let filename = format!("{}_{}.wav", species, next_number);
        
        // Add to metadata
        metadata.push(RecordingMetadata {
            id: id.clone(),
            url: url.clone(),
            common_name: common_name.clone(),
            scientific_name: scientific_name.clone(),
            filename,
            species,
            is_downloaded: false,
        });
    }
    
    Ok(metadata)
}

// Check which files already exist in the directory
fn update_download_status(
    metadata: &[RecordingMetadata],
    output_dir: &str
) -> Result<Vec<RecordingMetadata>, Box<dyn std::error::Error>> {
    let mut updated_metadata = metadata.to_vec();
    let dir = Path::new(output_dir);
    
    // Create a HashSet of existing filenames for quick lookup
    let existing_filenames: HashSet<String> = updated_metadata.iter()
        .map(|m| m.filename.clone())
        .collect();
    
    // Also track IDs to prevent duplicates
    let existing_ids: HashSet<String> = updated_metadata.iter()
        .map(|m| m.id.clone())
        .collect();
    
    // Check for existing files
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if let Some(ext) = path.extension() {
            if ext == "wav" {
                if let Some(filename) = path.file_name() {
                    let filename_str = filename.to_string_lossy().to_string();
                    
                    // Check if this file is already in our metadata
                    if existing_filenames.contains(&filename_str) {
                        // File exists in metadata, mark as downloaded
                        for meta in &mut updated_metadata {
                            if meta.filename == filename_str {
                                meta.is_downloaded = true;
                                println!("Found existing file: {}", filename_str);
                                break;
                            }
                        }
                    } else {
                        // File exists on disk but not in metadata - add it
                        println!("Found file not in metadata: {} - adding to metadata", filename_str);
                        
                        // Try to extract species from filename (format should be species_number.wav)
                        let species = if let Some(underscore_pos) = filename_str.rfind('_') {
                            filename_str[0..underscore_pos].to_string()
                        } else {
                            // Can't parse, use filename without extension as species
                            if let Some(dot_pos) = filename_str.rfind('.') {
                                filename_str[0..dot_pos].to_string()
                            } else {
                                filename_str.clone()
                            }
                        };
                        
                        // Generate placeholder data for the new entry
                        let common_name = species.replace('_', " ");
                        
                        // Create unique ID that won't conflict with existing IDs
                        let mut unique_id = format!("local_{}", filename_str.replace('.', "_"));
                        let mut counter = 1;
                        while existing_ids.contains(&unique_id) {
                            unique_id = format!("local_{}_{}", filename_str.replace('.', "_"), counter);
                            counter += 1;
                        }
                        
                        // Create a new metadata entry for this file
                        updated_metadata.push(RecordingMetadata {
                            id: unique_id,
                            url: "file://local".to_string(), // Placeholder URL
                            common_name: common_name.clone(),
                            scientific_name: "Unknown".to_string(),
                            filename: filename_str,
                            species,
                            is_downloaded: true, // Mark as downloaded since it exists
                        });
                    }
                }
            }
        }
    }
    
    // Count how many are already downloaded
    let downloaded_count = updated_metadata.iter().filter(|m| m.is_downloaded).count();
    let added_count = updated_metadata.len() - metadata.len();
    
    println!("Found {} files already downloaded out of {} total", 
             downloaded_count, updated_metadata.len());
    
    if added_count > 0 {
        println!("Added {} new files found on disk to metadata", added_count);
    }
    
    Ok(updated_metadata)
}

// Save metadata to CSV
fn write_metadata_csv(
    metadata_path: &Path,
    metadata: &[RecordingMetadata]
) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = csv::Writer::from_path(metadata_path)?;
    
    // Write header
    writer.write_record(&[
        "filename", "species", "original_url", "id", "common_name", "scientific_name", "is_downloaded"
    ])?;
    
    // Write data
    for meta in metadata {
        writer.write_record(&[
            &meta.filename,
            &meta.species,
            &meta.url,
            &meta.id,
            &meta.common_name,
            &meta.scientific_name,
            &meta.is_downloaded.to_string(),
        ])?;
    }
    
    writer.flush()?;
    println!("Metadata saved to {}", metadata_path.display());
    Ok(())
}

// Download missing files
fn download_missing_files(
    client: &Client,
    metadata: &[RecordingMetadata],
    output_dir: &str,
    download_delay_ms: u64
) -> Result<(), Box<dyn std::error::Error>> {
    // Count how many files need to be downloaded
    let to_download = metadata.iter().filter(|m| !m.is_downloaded).count();
    println!("Need to download {} files", to_download);
    
    if to_download == 0 {
        println!("No new files to download!");
        return Ok(());
    }
    
    // Create thread pool for downloads
    let pool = ThreadPool::new(3);
    let last_request_time = Arc::new(Mutex::new(Instant::now()));
    let metadata_path = Path::new(output_dir).join("metadata.csv");
    
    // Mutex to track successfully downloaded files for updating metadata
    let downloaded_ids = Arc::new(Mutex::new(Vec::new()));
    
    // Start downloads for non-downloaded files
    for meta in metadata.iter().filter(|m| !m.is_downloaded) {
        let client = client.clone();
        let url = meta.url.clone();
        let id = meta.id.clone();
        let filename = meta.filename.clone();
        let output_dir = output_dir.to_string();
        let download_delay = Duration::from_millis(download_delay_ms);
        let last_request_time = Arc::clone(&last_request_time);
        let downloaded_ids = Arc::clone(&downloaded_ids);
        
        pool.execute(move || {
            // Rate limiting within thread
            {
                let mut last_time = last_request_time.lock().unwrap();
                let now = Instant::now();
                let time_since_last = now.duration_since(*last_time);
                
                if time_since_last < download_delay {
                    // Sleep for the remaining time
                    let sleep_time = download_delay - time_since_last;
                    thread::sleep(sleep_time);
                }
                
                // Update the last request time
                *last_time = Instant::now();
            }
            
            // Download file
            let mp3_path = Path::new(&output_dir).join(filename.replace(".wav", ".mp3"));
            let wav_path = Path::new(&output_dir).join(&filename);
            
            println!("Downloading: {} -> {}", url, mp3_path.display());
            
            // Download with retries
            let max_retries = 3;
            let mut retry_count = 0;
            let mut retry_delay = Duration::from_millis(1000);
            
            while retry_count <= max_retries {
                match client.get(&url).send() {
                    Ok(response) => {
                        if !response.status().is_success() {
                            println!("Download failed with status: {}", response.status());
                        } else {
                            match response.bytes() {
                                Ok(bytes) => {
                                    // Save to MP3 file
                                    match File::create(&mp3_path) {
                                        Ok(mut file) => {
                                            if let Err(e) = file.write_all(&bytes) {
                                                println!("Error writing file: {}", e);
                                            } else {
                                                // Convert to WAV
                                                match convert_using_ffmpeg(&mp3_path, &wav_path) {
                                                    Ok(_) => {
                                                        println!("Successfully downloaded and converted: {}", filename);
                                                        
                                                        // Add to successful downloads
                                                        let mut ids = downloaded_ids.lock().unwrap();
                                                        ids.push(id.clone());
                                                        
                                                        break;
                                                    },
                                                    Err(e) => {
                                                        println!("Error converting to WAV: {}", e);
                                                    }
                                                }
                                            }
                                        },
                                        Err(e) => {
                                            println!("Error creating file: {}", e);
                                        }
                                    }
                                },
                                Err(e) => {
                                    println!("Error downloading bytes: {}", e);
                                }
                            }
                        }
                    },
                    Err(e) => {
                        println!("Download error: {}", e);
                    }
                }
                
                retry_count += 1;
                if retry_count <= max_retries {
                    println!("Retrying in {:?}...", retry_delay);
                    thread::sleep(retry_delay);
                    retry_delay *= 2; // Exponential backoff
                } else {
                    println!("Failed to download {} after {} attempts", url, max_retries + 1);
                }
            }
        });
    }
    
    // Wait for all downloads to complete
    pool.join();
    
    // Update metadata with successful downloads
    let successful_ids = downloaded_ids.lock().unwrap();
    if !successful_ids.is_empty() {
        let mut updated_metadata = metadata.to_vec();
        for id in successful_ids.iter() {
            for meta in &mut updated_metadata {
                if &meta.id == id {
                    meta.is_downloaded = true;
                    break;
                }
            }
        }
        
        // Save updated metadata
        write_metadata_csv(&metadata_path, &updated_metadata)?;
        println!("Updated metadata with {} newly downloaded files", successful_ids.len());
    }
    
    Ok(())
}

// Convert common name to normalized species name
fn format_species_name(name: &str) -> String {
    name.to_lowercase().replace(' ', "_")
}

// Convert MP3 to WAV using ffmpeg
fn convert_using_ffmpeg(mp3_path: &Path, wav_path: &Path) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use std::process::Command;
    
    println!("Converting with ffmpeg: {} → {}", mp3_path.display(), wav_path.display());
    
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
        Ok(())
    } else {
        let error = String::from_utf8_lossy(&output.stderr);
        Err(format!("ffmpeg conversion failed: {}", error).into())
    }
}

// Batch convert directory function
fn batch_convert_directory(dir_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let dir = Path::new(dir_path);
    if !dir.is_dir() {
        return Err(format!("{} is not a directory", dir_path).into());
    }
    
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension().map_or(false, |ext| ext == "mp3") {
            let wav_path = path.with_extension("wav");
            
            if !wav_path.exists() {
                println!("Converting: {}", path.display());
                match convert_using_ffmpeg(&path, &wav_path) {
                    Ok(_) => println!("Conversion successful: {}", wav_path.display()),
                    Err(e) => println!("Error converting {}: {}", path.display(), e),
                }
            } else {
                println!("Skipping: {} (WAV already exists)", path.display());
            }
        }
    }
    
    Ok(())
}

// Add this new function to load just the existing metadata without adding new entries
fn load_existing_metadata(
    metadata_path: &Path
) -> Result<Vec<RecordingMetadata>, Box<dyn std::error::Error>> {
    let mut metadata = Vec::new();
    
    if metadata_path.exists() {
        let mut reader = csv::Reader::from_path(metadata_path)?;
        
        for result in reader.records() {
            let record = result?;
            if record.len() >= 6 { // Basic validation
                let filename = record[0].to_string();
                let species = record[1].to_string();
                let url = record[2].to_string();
                let id = record[3].to_string();
                let common_name = record[4].to_string();
                let scientific_name = record[5].to_string();
                // Parse is_downloaded flag if it exists
                let is_downloaded = if record.len() >= 7 {
                    record[6].parse::<bool>().unwrap_or(false)
                } else {
                    false
                };
                
                metadata.push(RecordingMetadata {
                    id,
                    url,
                    common_name,
                    scientific_name,
                    filename,
                    species,
                    is_downloaded,
                });
            }
        }
    } else {
        return Err("Metadata file not found".into());
    }
    
    println!("Loaded {} entries from existing metadata", metadata.len());
    Ok(metadata)
}