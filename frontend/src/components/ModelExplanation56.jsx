import React, { useState } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Divider,
  Alert,
  AlertTitle,
  ToggleButtonGroup,
  ToggleButton,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import config from '../resources/config.json';

const ExplanationTab = ({ predictionResult, predictionFeatures }) => {
  const [explanationMethod, setExplanationMethod] = useState('shap');
  const [modelType, setModelType] = useState('base');
  const [plotType, setPlotType] = useState('waterfall');
  const [explanationType, setExplanationType] = useState('global');
  const [classIndex, setClassIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [plotData, setPlotData] = useState(null);
  const [featureValues, setFeatureValues] = useState(null);

  const plotTypes = {
    shap: explanationType === 'global' 
      ? ['beeswarm', 'bar'] 
      : ['waterfall', 'force'],
    lime: ['bar', 'explanation']
  };

  const formatValue = (value) => {
    const absValue = Math.abs(value);
    if (absValue < 0.01) {
      return value.toExponential(2);
    }
    return value.toFixed(4);
  };

  const getFilteredFeatures = () => {
    // Generate sequential array of feature indices based on model type
    return modelType === 'base' 
      ? Array.from({ length: 36 }, (_, i) => i)  // [0,1,2,...,35] for base model
      : Array.from({ length: 19 }, (_, i) => i);  // [0,1,2,...,18] for unbiased model
  };

  const handleExplanationRequest = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const modelTypeParam = modelType === 'base' ? 'best_model' : 'unbiased';
      const features = getFilteredFeatures();
      
      if (explanationType === 'global') {
        // Fetch global explanation
        const url = explanationMethod === 'shap'
          ? `${config.API_BASE_URL}/explain/global/summary-plot?model_type=${modelTypeParam}&plot_type=${plotType}&max_display=${modelType === 'base' ? 36 : 19}`
          : `${config.API_BASE_URL}/lime/global/plot`;

        // Get plot
        const plotResponse = await fetch(url, {
          method: explanationMethod === 'lime' ? 'POST' : 'GET',
          ...(explanationMethod === 'lime' && {
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              model_type: modelTypeParam,
              num_features: modelType === 'base' ? 36 : 19,
              num_samples: 5000
            })
          })
        });

        if (!plotResponse.ok) throw new Error('Failed to fetch explanation plot');
        
        // Handle plot response
        const contentType = plotResponse.headers.get('content-type');
        if (contentType && contentType.includes('image/png')) {
          const blob = await plotResponse.blob();
          setPlotData(URL.createObjectURL(blob));
        } else {
          const data = await plotResponse.json();
          if (data.image) {
            setPlotData(`data:image/png;base64,${data.image}`);
          }
        }

        // Get feature importance values
        if (explanationMethod === 'shap') {
          const valuesResponse = await fetch(`${config.API_BASE_URL}/explain/global/feature-importance?model_type=${modelTypeParam}`);
          if (!valuesResponse.ok) throw new Error('Failed to fetch SHAP values');
          const valuesData = await valuesResponse.json();
          setFeatureValues({
            type: 'shap',
            values: valuesData.feature_importance
          });
        } else {
          const valuesResponse = await fetch(`${config.API_BASE_URL}/lime/global/explanation`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              model_type: modelTypeParam,
              num_features: modelType === 'base' ? 36 : 19,
              num_samples: 5000
            })
          });
          if (!valuesResponse.ok) throw new Error('Failed to fetch LIME values');
          const valuesData = await valuesResponse.json();
          setFeatureValues({
            type: 'lime',
            values: Object.fromEntries(
              valuesData.global_explanations.map(item => [
                item.feature,
                item.importance
              ])
            )
          });
        }
      } else {
        // Local explanation
        try {
          if (explanationMethod === 'shap') {
            // Get SHAP waterfall plot and values
            const response = await fetch(`${config.API_BASE_URL}/explain/local/waterfall`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                features,
                model_type: modelTypeParam,
                max_display: modelType === 'base' ? 36 : 19,
                class_index: classIndex
              }),
            });

            if (!response.ok) throw new Error('Failed to fetch SHAP explanation');
            const data = await response.json();
            
            if (data.image) {
              setPlotData(`data:image/png;base64,${data.image}`);
              setFeatureValues({
                type: 'shap',
                values: data.shap_values.feature_values,
                baseValue: data.shap_values.base_value
              });
            }
          } else {
            // Get LIME plot
            const plotResponse = await fetch(`${config.API_BASE_URL}/lime/local/plot`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                features,
                model_type: modelTypeParam,
                num_features: modelType === 'base' ? 36 : 19,
                num_samples: 5000
              }),
            });

            if (!plotResponse.ok) throw new Error('Failed to fetch LIME plot');
            const blob = await plotResponse.blob();
            setPlotData(URL.createObjectURL(blob));

            // Get LIME values
            const valuesResponse = await fetch(`${config.API_BASE_URL}/lime/local/explanation`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                features,
                model_type: modelTypeParam,
                num_features: modelType === 'base' ? 36 : 19,
                num_samples: 5000
              }),
            });

            if (!valuesResponse.ok) throw new Error('Failed to fetch LIME values');
            const valuesData = await valuesResponse.json();
            setFeatureValues({
              type: 'lime',
              values: Object.fromEntries(
                valuesData.explanations.map(item => [
                  item.feature,
                  item.importance
                ])
              ),
              prediction: valuesData.prediction
            });
          }
        } catch (err) {
          console.error('Local explanation error:', err);
          setError(err.message);
        }
      }
    } catch (err) {
      console.error('Explanation error:', err);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Box sx={{ width: '100%' }}>
      
      <Paper sx={{ p: 3, mb: 4 }}>
        <Typography variant="body1" paragraph>
          This section provides both global and local explanations using SHAP (SHapley Additive exPlanations) 
          and LIME (Local Interpretable Model-agnostic Explanations) values.
        </Typography>

        <Box sx={{ mb: 3, mt: 3 }}>
          <Grid container spacing={3} alignItems="center">
            {/* Model Type */}
            <Grid item xs={12} sm={6} md={2}>
              <ToggleButtonGroup
                color="primary"
                value={modelType}
                exclusive
                onChange={(_, value) => value && setModelType(value)}
                fullWidth
                size="small"
              >
                <ToggleButton value="base">Base</ToggleButton>
                <ToggleButton value="unbiased">Unbiased</ToggleButton>
              </ToggleButtonGroup>
            </Grid>

            <Grid item xs={12} sm={6} md={2}>
              <ToggleButtonGroup
                color="primary"
                value={explanationMethod}
                exclusive
                onChange={(_, value) => value && setExplanationMethod(value)}
                fullWidth
                size="small"
              >
                <ToggleButton value="shap">SHAP</ToggleButton>
                <ToggleButton value="lime">LIME</ToggleButton>
              </ToggleButtonGroup>
            </Grid>

            {/* Explanation Type */}
            <Grid item xs={12} sm={6} md={2}>
              <ToggleButtonGroup
                color="primary"
                value={explanationType}
                exclusive
                onChange={(_, value) => value && setExplanationType(value)}
                fullWidth
                size="small"
              >
                <ToggleButton value="global">Global</ToggleButton>
                <ToggleButton value="local">Local</ToggleButton>
              </ToggleButtonGroup>
            </Grid>

            {/* Plot Type */}
            <Grid item xs={12} sm={6} md={3}>
              <ToggleButtonGroup
                color="primary"
                value={plotType}
                exclusive
                onChange={(_, value) => value && setPlotType(value)}
                fullWidth
                size="small"
              >
                {plotTypes[explanationMethod].map((type) => (
                  <ToggleButton key={type} value={type}>
                    {type.charAt(0).toUpperCase() + type.slice(1)}
                  </ToggleButton>
                ))}
              </ToggleButtonGroup>
            </Grid>
            
            {/* Class Selection - Only show for local explanations */}
            {explanationType === 'local' && (
              <Grid item xs={12} sm={6} md={3}>
                <Typography variant="body2" sx={{ mb: 1 }}>Target Class</Typography>
                <ToggleButtonGroup
                  color="primary"
                  value={classIndex}
                  exclusive
                  onChange={(_, value) => value !== null && setClassIndex(value)}
                  fullWidth
                  size="small"
                >
                  <ToggleButton value={2}>Enrolled (2)</ToggleButton>
                  <ToggleButton value={1}>Dropout (1)</ToggleButton>
                  <ToggleButton value={0}>Graduate (0)</ToggleButton>
                </ToggleButtonGroup>
              </Grid>
            )}

            {/* Generate Button */}
            <Grid item xs={12} md={explanationType === 'local' ? 2 : 3}>
              <Box>
                <Button
                  variant="contained"
                  fullWidth
                  onClick={handleExplanationRequest}
                  sx={{
                    background: 'linear-gradient(135deg,rgb(89, 104, 172) 0%,rgb(72, 82, 218) 100%)',
                    py: 2,
                  }}
                >
                  {isLoading ? <CircularProgress size={24} color="inherit" /> : 'Generate'}
                </Button>
              </Box>
            </Grid>
          </Grid>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        

        {/* Visualization Panel with Guide */}
        {plotData && (
          <Box sx={{ mt: 3, display: 'flex', gap: 3 }}>
            {/* Left Panel - Plot */}
            <Box sx={{ flex: '1 1 60%' }}>
              {/* Plot */}
              <Box 
                component="img" 
                src={plotData}
                alt="Model Explanation Visualization"
                sx={{
                  width: '100%',
                  height: 'auto',
                  objectFit: 'contain',
                  mb: 3
                }}
              />
            </Box>

            {/* Right Panel - Guide and Values */}
            <Box sx={{ flex: '1 1 40%', minWidth: '250px', display: 'flex', flexDirection: 'column', gap: 3 }}>
              {/* Interpretation Guide */}
              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Interpretation Guide
                </Typography>
                <Divider sx={{ my: 2 }} />
                
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    SHAP Explanations
                  </Typography>
                  <Typography variant="body2" component="ul" sx={{ pl: 2 }}>
                    <li>Bar Plot: Shows average impact of each feature on model output</li>
                    <li>Waterfall Plot: Shows how each feature contributes to a specific prediction</li>
                    <li>Force Plot: Visualizes the push and pull of features on the model output</li>
                    <li>Positive values (green) indicate the feature increases the prediction</li>
                    <li>Negative values (red) indicate the feature decreases the prediction</li>
                  </Typography>
                </Box>
                
                <Box>
                  <Typography variant="subtitle1" gutterBottom>
                    LIME Explanations
                  </Typography>
                  <Typography variant="body2" component="ul" sx={{ pl: 2 }}>
                    <li>Creates interpretable model locally around predictions</li>
                    <li>Shows feature contributions with confidence intervals</li>
                    <li>Bar Plot: Shows feature importance scores</li>
                    <li>Explanation Plot: Detailed view of how features affect the prediction</li>
                    <li>Green values show features supporting the prediction</li>
                    <li>Red values show features contradicting the prediction</li>
                  </Typography>
                </Box>
              </Paper>

              {/* Values Table */}
              {featureValues && (
                <Paper sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Feature {featureValues.type === 'shap' ? 'SHAP' : 'LIME'} Values
                  </Typography>
                  <TableContainer sx={{ maxHeight: '400px' }}>
                    <Table size="small" stickyHeader>
                      <TableHead>
                        <TableRow>
                          <TableCell>Feature</TableCell>
                          <TableCell align="right">Value</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {Object.entries(featureValues.values)
                          .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                          .map(([feature, value]) => (
                            <TableRow
                              key={feature}
                              sx={{
                                backgroundColor: value > 0 ? 'rgba(76, 175, 80, 0.1)' : 'rgba(244, 67, 54, 0.1)'
                              }}
                            >
                              <TableCell component="th" scope="row">
                                {feature}
                              </TableCell>
                              <TableCell 
                                align="right"
                                sx={{ 
                                  color: value > 0 ? 'success.main' : 'error.main',
                                  fontWeight: 'medium'
                                }}
                              >
                                {formatValue(value)}
                              </TableCell>
                            </TableRow>
                        ))}
                        {featureValues.baseValue !== undefined && (
                          <TableRow sx={{ backgroundColor: 'action.hover' }}>
                            <TableCell><strong>Base Value</strong></TableCell>
                            <TableCell align="right">
                              <strong>{formatValue(featureValues.baseValue)}</strong>
                            </TableCell>
                          </TableRow>
                        )}
                        {featureValues.prediction && (
                          <TableRow sx={{ backgroundColor: 'action.hover' }}>
                            <TableCell><strong>Predicted Class</strong></TableCell>
                            <TableCell align="right">
                              <strong>{featureValues.prediction.class_name}</strong>
                            </TableCell>
                          </TableRow>
                        )}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Paper>

              )}
            </Box>
      
          </Box>
        )}
      </Paper>
    </Box>
  );
};

export default ExplanationTab;