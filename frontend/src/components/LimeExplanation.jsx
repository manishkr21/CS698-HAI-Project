import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Typography,
  Stack,
  Button,
  TextField,
  Slider,
  Tabs,
  Tab,
  LinearProgress,
  Alert,
  Snackbar
} from '@mui/material';
import { styled } from '@mui/material/styles';
import config from '../resources/config.json';

const StyledCard = styled(Card)(({ theme }) => ({
  margin: theme.spacing(2),
  padding: theme.spacing(2)
}));

const StyledBox = styled(Box)(({ theme }) => ({
  marginBottom: theme.spacing(2)
}));

const FeatureBar = styled(Box)(({ theme }) => ({
  background: theme.palette.primary.light,
  height: 20,
  borderRadius: theme.shape.borderRadius,
  position: 'relative'
}));

const FeatureProgress = styled(Box)(({ theme }) => ({
  background: theme.palette.primary.main,
  height: '100%',
  borderRadius: theme.shape.borderRadius,
  transition: 'width 0.5s ease-in-out'
}));

const LimeExplanation = ({
  selectedModel,
  features,
  featureNames,
  prediction,
  classNames
}) => {
  const [activeTab, setActiveTab] = useState('local');
  const [numFeatures, setNumFeatures] = useState(10);
  const [numSamples, setNumSamples] = useState(5000);
  const [sampleSize, setSampleSize] = useState(100);
  const [localPlot, setLocalPlot] = useState(null);
  const [globalPlot, setGlobalPlot] = useState(null);
  const [featureImportance, setFeatureImportance] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [predictionInfo, setPredictionInfo] = useState(null);
  const [tabValue, setTabValue] = useState(0);
  const [showToast, setShowToast] = useState(false);
  const [toastMessage, setToastMessage] = useState('');
  const [toastSeverity, setToastSeverity] = useState('error');

  const handleToast = (message, severity = 'error') => {
    setToastMessage(message);
    setToastSeverity(severity);
    setShowToast(true);
  };

  const fetchLocalExplanation = async () => {
    if (!features || !selectedModel) {
      handleToast('No prediction data available');
      return;
    }

    setIsLoading(true);
    try {
      // Fetch explanation data
      const dataResponse = await fetch(`${config.API_BASE_URL}/lime/local/explanation`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          features: features,
          model_type: selectedModel,
          num_features: numFeatures,
          num_samples: numSamples
        }),
      });

      if (!dataResponse.ok) {
        throw new Error(`HTTP error! status: ${dataResponse.status}`);
      }

      const explanationData = await dataResponse.json();
      setFeatureImportance(explanationData.explanations);
      setPredictionInfo(explanationData.prediction);

      // Fetch visualization
      const plotResponse = await fetch(`${config.API_BASE_URL}/lime/local/plot`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          features: features,
          model_type: selectedModel,
          num_features: numFeatures,
          num_samples: numSamples
        }),
      });

      if (!plotResponse.ok) {
        throw new Error(`HTTP error! status: ${plotResponse.status}`);
      }

      const blob = await plotResponse.blob();
      const imageUrl = URL.createObjectURL(blob);
      setLocalPlot(imageUrl);

    } catch (error) {
      console.error('LIME explanation error:', error);
      handleToast(error.message || 'Failed to fetch LIME explanation');
    }
    setIsLoading(false);
  };

  const fetchGlobalExplanation = async () => {
    setIsLoading(true);
    try {
      // Fetch global importance data
      const dataResponse = await fetch(`${config.API_BASE_URL}/lime/global/data`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          num_features: numFeatures,
          num_samples: numSamples,
          sample_size: sampleSize
        }),
      });

      if (!dataResponse.ok) {
        throw new Error(`HTTP error! status: ${dataResponse.status}`);
      }

      const importanceData = await dataResponse.json();
      setFeatureImportance(
        Object.entries(importanceData.feature_importance).map(([feature, importance]) => ({
          feature,
          importance
        }))
      );

      // Fetch visualization
      const plotResponse = await fetch(`${config.API_BASE_URL}/lime/global/plot`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          num_features: numFeatures,
          num_samples: numSamples,
          sample_size: sampleSize
        }),
      });

      if (!plotResponse.ok) {
        throw new Error(`HTTP error! status: ${plotResponse.status}`);
      }

      const blob = await plotResponse.blob();
      const imageUrl = URL.createObjectURL(blob);
      setGlobalPlot(imageUrl);

    } catch (error) {
      console.error('Global LIME explanation error:', error);
      handleToast(error.message || 'Failed to fetch global LIME explanation');
    }
    setIsLoading(false);
  };

  // Update display of prediction information
  const renderPredictionInfo = () => {
    const predInfo = predictionInfo || prediction;
    if (!predInfo) return null;
    
    return (
      <StyledBox>
        <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
          Current Prediction:
        </Typography>
        <Typography>
          Class: {predInfo.class_name} ({predInfo.class})
        </Typography>
        <Typography>
          Confidence: {(predInfo.probability * 100).toFixed(2)}%
        </Typography>
      </StyledBox>
    );
  };

  return (
    <StyledCard>
      <CardHeader title="LIME Model Explanations" />
      <CardContent>
        <Tabs
          value={tabValue}
          onChange={(_, newValue) => setTabValue(newValue)}
          variant="fullWidth"
          sx={{ mb: 2 }}
        >
          <Tab label="Local Explanation" />
          <Tab label="Global Explanation" />
        </Tabs>

        {tabValue === 0 && (
          <Stack spacing={3}>
            {renderPredictionInfo()}
            
            <StyledBox>
              <Typography gutterBottom>Number of Features to Show:</Typography>
              <TextField
                type="number"
                value={numFeatures}
                onChange={(e) => setNumFeatures(Math.max(5, Math.min(50, parseInt(e.target.value) || 5)))}
                InputProps={{ inputProps: { min: 5, max: 50 } }}
                fullWidth
                size="small"
              />
            </StyledBox>

            <StyledBox>
              <Typography gutterBottom>Number of Samples:</Typography>
              <Slider
                value={numSamples}
                min={1000}
                max={10000}
                step={1000}
                onChange={(_, value) => setNumSamples(value)}
                valueLabelDisplay="auto"
                marks
              />
              <Typography variant="caption" align="right">
                {numSamples}
              </Typography>
            </StyledBox>

            <Button
              variant="contained"
              onClick={fetchLocalExplanation}
              disabled={isLoading}
            >
              Generate Local Explanation
            </Button>

            {isLoading && <LinearProgress />}

            {localPlot && (
              <Box component="img" src={localPlot} alt="LIME Local Explanation" sx={{ width: '100%' }} />
            )}
          </Stack>
        )}

        {tabValue === 1 && (
          <Stack spacing={3}>
            <StyledBox>
              <Typography gutterBottom>Sample Size for Global Analysis:</Typography>
              <TextField
                type="number"
                value={sampleSize}
                onChange={(e) => setSampleSize(Math.max(50, Math.min(500, parseInt(e.target.value) || 50)))}
                InputProps={{ inputProps: { min: 50, max: 500, step: 50 } }}
                fullWidth
                size="small"
              />
            </StyledBox>

            <StyledBox>
              <Typography gutterBottom>Number of Features:</Typography>
              <TextField
                type="number"
                value={numFeatures}
                onChange={(e) => setNumFeatures(Math.max(5, Math.min(50, parseInt(e.target.value) || 5)))}
                InputProps={{ inputProps: { min: 5, max: 50 } }}
                fullWidth
                size="small"
              />
            </StyledBox>

            <StyledBox>
              <Typography gutterBottom>Samples per Instance:</Typography>
              <Slider
                value={numSamples}
                min={1000}
                max={10000}
                step={1000}
                onChange={(_, value) => setNumSamples(value)}
                valueLabelDisplay="auto"
                marks
              />
              <Typography variant="caption" align="right">
                {numSamples}
              </Typography>
            </StyledBox>

            <Button
              variant="contained"
              onClick={fetchGlobalExplanation}
              disabled={isLoading}
            >
              Generate Global Explanation
            </Button>

            {isLoading && <LinearProgress />}

            {globalPlot && (
              <Box component="img" src={globalPlot} alt="LIME Global Explanation" sx={{ width: '100%' }} />
            )}
          </Stack>
        )}

        {featureImportance && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" gutterBottom>
              Feature Importance Details
            </Typography>
            <Stack spacing={1}>
              {featureImportance.map(({ feature, importance }, idx) => (
                <Box key={idx} sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Typography sx={{ flex: 1 }}>{feature}</Typography>
                  <FeatureBar sx={{ flex: 2 }}>
                    <FeatureProgress sx={{ width: `${Math.abs(importance * 100)}%` }} />
                  </FeatureBar>
                  <Typography sx={{ width: 60, textAlign: 'right' }}>
                    {importance.toFixed(3)}
                  </Typography>
                </Box>
              ))}
            </Stack>
          </Box>
        )}

        <Snackbar
          open={showToast}
          autoHideDuration={5000}
          onClose={() => setShowToast(false)}
        >
          <Alert 
            onClose={() => setShowToast(false)} 
            severity={toastSeverity}
            sx={{ width: '100%' }}
          >
            {toastMessage}
          </Alert>
        </Snackbar>
      </CardContent>
    </StyledCard>
  );
};

export default LimeExplanation;