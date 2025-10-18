import React from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Grid,
  Typography,
  Avatar,
  Box,
  Button,
  Alert,
  AlertTitle,
  CircularProgress,
  Chip,
  LinearProgress,
  Stack
} from '@mui/material';
import { 
  Compare, 
  CompareArrows, 
  BarChart as BarChartIcon, 
  Balance, 
  CheckCircle, 
  Warning 
} from '@mui/icons-material';

export default function CompareModelsTab({ 
  comparisonResult, 
  predicting, 
  handleComparePredictions, 
  predictionForm 
}) {
  return (
    <Grid container spacing={3}>
      {/* Info Alert */}
      <Grid item xs={12}>
        <Alert severity="info" icon={<Compare />}>
          <AlertTitle>Model Comparison</AlertTitle>
          Compare predictions between baseline and bias-mitigated models to understand the impact of fairness interventions.
        </Alert>
      </Grid>

      {/* Compare Button Card */}
      <Grid item xs={12}>
        <Card>
          <CardHeader
            avatar={<Avatar sx={{ bgcolor: 'secondary.main' }}><CompareArrows /></Avatar>}
            title="Compare Model Predictions"
            titleTypographyProps={{ variant: 'h6' }}
          />
          <CardContent>
            <Button
              fullWidth
              variant="contained"
              size="large"
              onClick={handleComparePredictions}
              startIcon={predicting ? <CircularProgress size={20} color="inherit" /> : <Compare />}
              disabled={predicting || Object.keys(predictionForm).length === 0}
              sx={{
                background: 'linear-gradient(45deg, #1976d2 30%, #2e7d32 90%)',
                py: 2,
              }}
            >
              {predicting ? 'Comparing...' : 'Compare Both Models'}
            </Button>
          </CardContent>
        </Card>
      </Grid>

      {/* Comparison Results */}
      {comparisonResult && (
        <>
          {/* Baseline Model */}
          <Grid item xs={12} md={6}>
            <Card sx={{ height: '100%' }}>
              <CardHeader
                avatar={<Avatar sx={{ bgcolor: 'primary.main' }}><BarChartIcon /></Avatar>}
                title="Baseline Model"
                subheader="Prediction without bias mitigation"
                titleTypographyProps={{ variant: 'h6' }}
              />
              <CardContent>
                {comparisonResult.comparison.baseline.error ? (
                  <Alert severity="error">{comparisonResult.comparison.baseline.error}</Alert>
                ) : (
                  <Stack spacing={3}>
                    <Card variant="outlined" sx={{ p: 3, textAlign: 'center', bgcolor: 'primary.50' }}>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        PREDICTION
                      </Typography>
                      <Typography variant="h3" color="primary.main" fontWeight="bold">
                        {comparisonResult.comparison.baseline.prediction_label}
                      </Typography>
                    </Card>
                    
                    {comparisonResult.comparison.baseline.probabilities && (
                      <Box>
                        <Typography variant="subtitle1" fontWeight="600" gutterBottom>
                          Confidence Levels
                        </Typography>
                        <Stack spacing={2}>
                          {Object.entries(comparisonResult.comparison.baseline.probabilities).map(([label, prob]) => (
                            <Box key={label}>
                              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                                <Typography variant="body2" fontWeight="500">{label}</Typography>
                                <Typography variant="body2" fontWeight="bold" color="primary.main">
                                  {(prob * 100).toFixed(1)}%
                                </Typography>
                              </Box>
                              <LinearProgress 
                                variant="determinate" 
                                value={prob * 100}
                                sx={{
                                  height: 8,
                                  borderRadius: 4,
                                  bgcolor: 'grey.200',
                                  '& .MuiLinearProgress-bar': {
                                    borderRadius: 4,
                                    bgcolor: 'primary.main'
                                  }
                                }}
                              />
                            </Box>
                          ))}
                        </Stack>
                      </Box>
                    )}
                  </Stack>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Mitigated Model */}
          <Grid item xs={12} md={6}>
            <Card sx={{ height: '100%' }}>
              <CardHeader
                avatar={<Avatar sx={{ bgcolor: 'success.main' }}><Balance /></Avatar>}
                title="Bias-Mitigated Model"
                subheader="Prediction with fairness intervention"
                titleTypographyProps={{ variant: 'h6' }}
              />
              <CardContent>
                {comparisonResult.comparison.mitigated.error ? (
                  <Alert severity="warning">{comparisonResult.comparison.mitigated.error}</Alert>
                ) : (
                  <Stack spacing={3}>
                    <Card variant="outlined" sx={{ p: 3, textAlign: 'center', bgcolor: 'success.50' }}>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        PREDICTION
                      </Typography>
                      <Typography variant="h3" color="success.main" fontWeight="bold">
                        {comparisonResult.comparison.mitigated.prediction_label}
                      </Typography>
                    </Card>
                    
                    {comparisonResult.comparison.mitigated.probabilities && (
                      <Box>
                        <Typography variant="subtitle1" fontWeight="600" gutterBottom>
                          Confidence Levels
                        </Typography>
                        <Stack spacing={2}>
                          {Object.entries(comparisonResult.comparison.mitigated.probabilities).map(([label, prob]) => (
                            <Box key={label}>
                              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                                <Typography variant="body2" fontWeight="500">{label}</Typography>
                                <Typography variant="body2" fontWeight="bold" color="success.main">
                                  {(prob * 100).toFixed(1)}%
                                </Typography>
                              </Box>
                              <LinearProgress 
                                variant="determinate" 
                                value={prob * 100} 
                                color="success"
                                sx={{
                                  height: 8,
                                  borderRadius: 4,
                                  bgcolor: 'grey.200',
                                  '& .MuiLinearProgress-bar': {
                                    borderRadius: 4
                                  }
                                }}
                              />
                            </Box>
                          ))}
                        </Stack>
                      </Box>
                    )}
                  </Stack>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Match/Differ Alert */}
          <Grid item xs={12}>
            <Card elevation={3}>
              <CardContent sx={{ p: 3 }}>
                {comparisonResult.predictions_match ? (
                  <Alert 
                    severity="success" 
                    icon={<CheckCircle />}
                    sx={{ 
                      fontSize: '1rem',
                      '& .MuiAlert-icon': { fontSize: 28 }
                    }}
                  >
                    <AlertTitle sx={{ fontSize: '1.1rem', fontWeight: 600 }}>
                      Predictions Match
                    </AlertTitle>
                    Both models agree on the prediction: <strong>{comparisonResult.comparison.baseline.prediction_label}</strong>
                  </Alert>
                ) : (
                  <Alert 
                    severity="warning" 
                    icon={<Warning />}
                    sx={{ 
                      fontSize: '1rem',
                      '& .MuiAlert-icon': { fontSize: 28 }
                    }}
                  >
                    <AlertTitle sx={{ fontSize: '1.1rem', fontWeight: 600 }}>
                      Predictions Differ
                    </AlertTitle>
                    <Box component="span">
                      Baseline: <Chip label={comparisonResult.comparison.baseline.prediction_label} color="primary" size="small" sx={{ mx: 0.5 }} />
                      {' vs '}
                      Mitigated: <Chip label={comparisonResult.comparison.mitigated.prediction_label} color="success" size="small" sx={{ mx: 0.5 }} />
                    </Box>
                  </Alert>
                )}
              </CardContent>
            </Card>
          </Grid>
        </>
      )}
    </Grid>
  );
}
