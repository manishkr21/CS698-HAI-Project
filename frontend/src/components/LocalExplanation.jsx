import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Typography,
  Stack,
  Button,
  LinearProgress,
  Alert,
  Grid,
  ToggleButtonGroup,
  ToggleButton,
  Slider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper
} from '@mui/material';
import { styled } from '@mui/material/styles';
import config from '../resources/config.json';

const StyledCard = styled(Card)(({ theme }) => ({
  margin: theme.spacing(2),
  padding: theme.spacing(2)
}));

const LocalExplanation = ({
  selectedModel,
  features,
  featureNames,
  prediction,
  classNames
}) => {
  const [selectedClass, setSelectedClass] = useState(0);
  const [numFeatures, setNumFeatures] = useState(20);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [explanationPlot, setExplanationPlot] = useState(null);
  const [shapValues, setShapValues] = useState(null);
  const [limeValues, setLimeValues] = useState(null);
  const [myclassname] = useState(["Enrolled", "Dropout", "Graduate"]);
  const [explanationType, setExplanationType] = useState('shap');

  const handleFetchExplanation = async () => {
    setIsLoading(true);
    setError(null);
    try {
      if (explanationType === 'shap') {
        const response = await fetch(`${config.API_BASE_URL}/explain/local/waterfall`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            features,
            model_type: selectedModel,
            max_display: numFeatures,
            class_index: selectedClass
          }),
        });

        if (!response.ok) {
          throw new Error('Failed to fetch SHAP explanation');
        }

        const data = await response.json();
        
        if (data.image) {
          setExplanationPlot(`data:image/png;base64,${data.image}`);
        }
        if (data.shap_values) {
          setShapValues(data.shap_values);
          setLimeValues(null);
        }
      } else {
        // Fetch LIME explanation
        const plotResponse = await fetch(`${config.API_BASE_URL}/lime/local/plot`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            features,
            model_type: selectedModel,
            num_features: numFeatures,
            num_samples: 5000
          }),
        });

        if (!plotResponse.ok) {
          throw new Error('Failed to fetch LIME plot');
        }

        const plotBlob = await plotResponse.blob();
        setExplanationPlot(URL.createObjectURL(plotBlob));

        // Fetch LIME values
        const valuesResponse = await fetch(`${config.API_BASE_URL}/lime/local/explanation`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            features,
            model_type: selectedModel,
            num_features: numFeatures,
            num_samples: 5000
          }),
        });

        if (!valuesResponse.ok) {
          throw new Error('Failed to fetch LIME values');
        }

        const valuesData = await valuesResponse.json();
        setLimeValues(valuesData);
        setShapValues(null);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  // Function to format SHAP values for display
  const formatShapValue = (value) => {
    const absValue = Math.abs(value);
    if (absValue < 0.01) {
      return value.toExponential(2);
    }
    return value.toFixed(4);
  };

  // Function to format LIME importance values
  const formatLimeValue = (value) => {
    return value.toFixed(4);
  };

  return (
    <StyledCard>
      <CardHeader title="Local Model Explanation" sx={{maxWidth: 400, mx: 'auto', mb: 2}} />
      <CardContent>

        <Grid sx={{width: 950, mx: 'auto', mb: 4}}>

        <Grid container spacing={12}>
            {/* Explanation Type Toggle */}
            <Grid item xs={12} sm={6} md={3}>
              <Typography gutterBottom>Explanation Method</Typography>
              <ToggleButtonGroup
                color="primary"
                value={explanationType}
                exclusive
                onChange={(_, value) => value && setExplanationType(value)}
                fullWidth
                size="small"
              >
                <ToggleButton value="shap">SHAP</ToggleButton>
                <ToggleButton value="lime">LIME</ToggleButton>
              </ToggleButtonGroup>
            </Grid>

            {/* Target Class Toggle */}
            <Grid item xs={12} sm={6} md={3}>
              <Typography gutterBottom>Target Class</Typography>
              <ToggleButtonGroup
                color="primary"
                value={selectedClass}
                exclusive
                onChange={(_, value) => value !== null && setSelectedClass(value)}
                fullWidth
                size="small"
              >
                {myclassname?.map((className, idx) => (
                  <ToggleButton key={idx} value={idx}>
                    {className}
                  </ToggleButton>
                ))}
              </ToggleButtonGroup>
            </Grid>
            
            {/* Number of Features Slider */}
            <Grid item xs={12} sm={6} md={4}>
              <Typography gutterBottom>Number of Features</Typography>
              <Box sx={{ px: 2 }}>
                <Slider
                  value={numFeatures}
                  onChange={(_, value) => setNumFeatures(value)}
                  min={1}
                  max={36}
                  marks={[
                    { value: 1, label: '1' },
                    { value: 36, label: '36' }
                  ]}
                  valueLabelDisplay="auto"
                />
              </Box>
            </Grid>

            {/* Generate Button */}
            <Grid item xs={12} sm={6} md={2}>
              <Box sx={{ mt: 3 }}>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={handleFetchExplanation}
                  disabled={isLoading}
                  fullWidth
                >
                  {isLoading ? 'Generating...' : 'Generate'}
                </Button>
              </Box>
            </Grid>
          </Grid>

          {isLoading && <LinearProgress />}

          {error && (
            <Alert severity="error" onClose={() => setError(null)}>
              {error}
            </Alert>
          )}
        </Grid>

        <Stack spacing={3} >
         

          {(explanationPlot || shapValues || limeValues) && (
            <Grid container spacing={3}>
              {/* Plot Panel */}
              <Grid item xs={12} md={6} width={'65%'}>
                <Paper elevation={2} sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Feature Contributions Visualization
                  </Typography>
                  {explanationPlot && (
                    <Box
                      component="img"
                      src={explanationPlot}
                      alt="Local Explanation Plot"
                      sx={{
                        width: '100%',
                        height: 'auto',
                        maxWidth: '100%',
                        display: 'block',
                        margin: '0 auto'
                      }}
                    />
                  )}
                </Paper>
              </Grid>

              {/* Values Table */}
              <Grid item xs={12} md={6} width={'32%'}>
                <Paper elevation={2} sx={{ p: 2, height: '100%' }}>
                  <Typography variant="h6" gutterBottom>
                    Feature Contribution Values
                  </Typography>
                  {(shapValues || limeValues) && (
                    <TableContainer sx={{ maxHeight: '600px', overflow: 'auto' }}>
                      <Table size="small" stickyHeader>
                        <TableHead>
                          <TableRow>
                            <TableCell>Feature</TableCell>
                            <TableCell align="right">
                              {explanationType === 'shap' ? 'SHAP Value' : 'LIME Importance'}
                            </TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {shapValues && Object.entries(shapValues.feature_values).map(([feature, value]) => (
                            <TableRow
                              key={feature}
                              sx={{
                                backgroundColor: value > 0 ? 'rgba(76, 175, 80, 0.1)' : 'rgba(244, 67, 54, 0.1)'
                              }}
                            >
                              <TableCell component="th" scope="row">
                                {feature}
                              </TableCell>
                              <TableCell align="right" sx={{ color: value > 0 ? 'success.main' : 'error.main' }}>
                                {formatShapValue(value)}
                              </TableCell>
                            </TableRow>
                          ))}
                          {limeValues && limeValues.explanations.map((item) => (
                            <TableRow
                              key={item.feature}
                              sx={{
                                backgroundColor: item.importance > 0 ? 'rgba(76, 175, 80, 0.1)' : 'rgba(244, 67, 54, 0.1)'
                              }}
                            >
                              <TableCell component="th" scope="row">
                                {item.feature}
                              </TableCell>
                              <TableCell align="right" sx={{ color: item.importance > 0 ? 'success.main' : 'error.main' }}>
                                {formatLimeValue(item.importance)}
                              </TableCell>
                            </TableRow>
                          ))}
                          {shapValues && (
                            <TableRow sx={{ backgroundColor: 'action.hover' }}>
                              <TableCell><strong>Base Value</strong></TableCell>
                              <TableCell align="right">
                                <strong>{formatShapValue(shapValues.base_value)}</strong>
                              </TableCell>
                            </TableRow>
                          )}
                          {limeValues && (
                            <TableRow sx={{ backgroundColor: 'action.hover' }}>
                              <TableCell><strong>Predicted Class</strong></TableCell>
                              <TableCell align="right">
                                <strong>{limeValues.prediction.class_name}</strong>
                              </TableCell>
                            </TableRow>
                          )}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  )}
                 <Typography variant="body2" color="text.secondary" sx={{ mt: 4 }}>
                    {explanationType === 'shap'
                      ? 'Note: SHAP values represent the contribution of each feature to the prediction. Positive values indicate a push towards the predicted class, while negative values indicate a push away.'
                      : 'Note: LIME importance scores indicate how much each feature influenced the prediction. Positive scores suggest a positive influence towards the predicted class, while negative scores suggest a negative influence.'}
                  </Typography>
                </Paper>
                
              </Grid>
            </Grid>
          )}
        </Stack>
      </CardContent>
    </StyledCard>
  );
};

export default LocalExplanation;