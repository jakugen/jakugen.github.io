# Enhanced MFCC Feature Extraction

## Overview

This implementation enhances the original MFCC feature extraction with delta features, improved parameters, and better normalization for more reliable bird song classification.

## Key Improvements

### 1. Enhanced Feature Vector (13D → 60D)
- **Static MFCC**: 20 coefficients (increased from 13)
- **Delta MFCC**: 20 first-derivative features (temporal dynamics)
- **Delta-Delta MFCC**: 20 second-derivative features (acceleration)
- **Total**: 60-dimensional feature vector

### 2. Improved Signal Processing
- **Larger Filter Bank**: 40 Mel filters (increased from 26)
- **Cepstral Mean Normalization**: Removes channel effects
- **Better Windowing**: Optimized frame size and hop length

### 3. Temporal Modeling
- **Delta Features**: Capture how features change over time
- **Delta-Delta Features**: Capture acceleration of feature changes
- **Boundary Handling**: Proper padding for edge frames

## Usage

### Basic Feature Extraction
```javascript
import { extractEnhancedMfccFeatures } from './modules/mfcc.js';

// Extract 60-dimensional enhanced features
const features = await extractEnhancedMfccFeatures(audioBlob);
console.log(`Extracted ${features.length}D features`); // Should print "60D"
```

### Training with Enhanced Features
```javascript
import { trainModel } from './modules/training.js';

// Training automatically uses enhanced features
const { model, classMap } = await trainModel();
```

### Prediction with Enhanced Features
```javascript
// Prediction automatically uses enhanced features
const prediction = await predict(audioBlob, model);
```

## Feature Vector Structure

```
Enhanced MFCC Feature Vector (60D):
├── Static MFCC (0-19):     [c₀, c₁, c₂, ..., c₁₉]
├── Delta MFCC (20-39):     [Δc₀, Δc₁, Δc₂, ..., Δc₁₉]  
└── Delta-Delta MFCC (40-59): [ΔΔc₀, ΔΔc₁, ΔΔc₂, ..., ΔΔc₁₉]
```

## Technical Details

### Delta Feature Computation
Delta features are computed using finite differences:
```
Δc[t] = Σ(n=1 to N) n × (c[t+n] - c[t-n]) / Σ(n=1 to N) 2n²
```
Where N=2 (window size) and boundary conditions use padding.

### Cepstral Mean Normalization
Removes the mean of each coefficient across all frames:
```
c_normalized[t][i] = c[t][i] - mean(c[all_frames][i])
```

### Model Architecture Updates
The neural network input layer now expects 60-dimensional features:
```javascript
model.add(tf.layers.batchNormalization({inputShape: [60]}));
```

## Performance Expectations

### Accuracy Improvements
- **Expected**: 10-20% improvement in classification accuracy
- **Reason**: Better temporal modeling and increased feature richness

### Computational Cost
- **Feature Extraction**: ~2-3x slower (more computations)
- **Model Training**: ~4.6x more parameters in input layer
- **Memory Usage**: ~4.6x more feature storage

### Quality Metrics
- **Robustness**: Better handling of noise and variations
- **Consistency**: More stable predictions across similar recordings
- **Temporal Sensitivity**: Better capture of bird song dynamics

## Testing

Run the test suite to validate the implementation:
```bash
# Open in browser
open test-enhanced-mfcc.html
```

Or run unit tests programmatically:
```javascript
import { runMfccTests } from './modules/test-mfcc.js';
await runMfccTests();
```

## Backward Compatibility

The original `simpleExtractMfccFeatures()` function is preserved for backward compatibility but marked as deprecated. New implementations should use `extractEnhancedMfccFeatures()`.

## Migration Guide

### For Existing Models
1. **Retrain Required**: Existing models trained on 13D features won't work with 60D features
2. **Update Model Architecture**: Change input shape from [13] to [60]
3. **Update Feature Extraction**: Replace `simpleExtractMfccFeatures` with `extractEnhancedMfccFeatures`

### Code Changes
```javascript
// Old (13D features)
const features = await simpleExtractMfccFeatures(audioBlob);
model.add(tf.layers.dense({inputShape: [13], ...}));

// New (60D features)  
const features = await extractEnhancedMfccFeatures(audioBlob);
model.add(tf.layers.batchNormalization({inputShape: [60]}));
```

## Expected Results

With enhanced MFCC features, you should see:
- ✅ Higher classification accuracy (target: >85%)
- ✅ More consistent predictions for same species
- ✅ Better handling of background noise
- ✅ Improved temporal pattern recognition
- ✅ More robust performance across different audio conditions

## Next Steps

After implementing enhanced MFCC features, consider:
1. **Data Augmentation** (Task 3.1): Add audio augmentation techniques
2. **Advanced Model Architecture** (Task 4.1): Implement deeper networks
3. **Confidence Estimation** (Task 4.2): Add uncertainty quantification
4. **Multi-Feature Extraction** (Task 2.2): Add spectral and temporal features