import React, { useState } from 'react';
import {
  Card,
  CardHeader,
  CardContent,
  Avatar,
  Box,
  Typography,
  Chip,
  LinearProgress,
  Stack,
  Button,
  IconButton,
  Divider
} from '@mui/material';
import { CheckCircle, Psychology, HelpOutline, ThumbUp, ThumbDown } from '@mui/icons-material';

export default function PredictionResultCard({ predictionResult, onHelpClick, onFeedback }) {
  const [userFeedback, setUserFeedback] = useState(null); // 'positive', 'negative', or null

  const handleFeedback = (feedbackType) => {
    const newFeedback = userFeedback === feedbackType ? null : feedbackType;
    setUserFeedback(newFeedback);
    
    // Call parent callback if provided
    if (onFeedback) {
      onFeedback({
        prediction: predictionResult.prediction_label,
        feedback: newFeedback,
        probabilities: predictionResult.probabilities
      });
    }
  };

  return (
    <Card>
      <CardHeader
        avatar={<Avatar sx={{ bgcolor: 'success.main' }}><CheckCircle /></Avatar>}
        title="Prediction Result"
        titleTypographyProps={{ variant: 'h6' }}
      />
      <CardContent>
        {predictionResult ? (
          <Stack spacing={3}>
            <Card variant="outlined" sx={{ p: 3, textAlign: 'center', bgcolor: 'primary.50' }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                PREDICTION
              </Typography>
              <Typography variant="h4" color="primary.main" fontWeight="bold">
                {predictionResult.prediction_label}
              </Typography>
            </Card>
            
            {predictionResult.probabilities && (
              <Box>
                <Typography variant="h6" gutterBottom textAlign="center">
                  Confidence Levels
                </Typography>
                <Stack spacing={2}>
                  {Object.entries(predictionResult.probabilities)
                    .filter(([label, prob]) => prob > 0)
                    .map(([label, prob]) => (
                      <Box key={label}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                          <Typography variant="body2" fontWeight="medium">
                            {label}
                          </Typography>
                          <Typography variant="body2" color="primary.main" fontWeight="bold">
                            {(prob * 100).toFixed(1)}%
                          </Typography>
                        </Box>
                        <LinearProgress
                          variant="determinate"
                          value={prob * 100}
                          sx={{
                            height: 8,
                            borderRadius: 5,
                            bgcolor: 'grey.200',
                            '& .MuiLinearProgress-bar': {
                              borderRadius: 5,
                              background: 'linear-gradient(45deg, #1976d2 30%, #9c27b0 90%)',
                            },
                          }}
                        />
                      </Box>
                    ))}
                </Stack>
              </Box>
            )}

            {/* User Feedback Section */}
            <Box>
              <Divider sx={{ mb: 2 }} />
              <Typography variant="body2" color="text.secondary" textAlign="center" gutterBottom>
                Was this prediction helpful?
              </Typography>
              <Box sx={{ display: 'flex', justifyContent: 'center', gap: 1, mb: 2 }}>
                <IconButton
                  onClick={() => handleFeedback('positive')}
                  sx={{
                    color: userFeedback === 'positive' ? 'success.main' : 'grey.500',
                    bgcolor: userFeedback === 'positive' ? 'success.50' : 'transparent',
                    border: userFeedback === 'positive' ? '2px solid' : '1px solid',
                    borderColor: userFeedback === 'positive' ? 'success.main' : 'grey.300',
                    '&:hover': {
                      bgcolor: 'success.50',
                      color: 'success.main',
                      borderColor: 'success.main',
                    },
                  }}
                >
                  <ThumbUp />
                </IconButton>
                <IconButton
                  onClick={() => handleFeedback('negative')}
                  sx={{
                    color: userFeedback === 'negative' ? 'error.main' : 'grey.500',
                    bgcolor: userFeedback === 'negative' ? 'error.50' : 'transparent',
                    border: userFeedback === 'negative' ? '2px solid' : '1px solid',
                    borderColor: userFeedback === 'negative' ? 'error.main' : 'grey.300',
                    '&:hover': {
                      bgcolor: 'error.50',
                      color: 'error.main',
                      borderColor: 'error.main',
                    },
                  }}
                >
                  <ThumbDown />
                </IconButton>
              </Box>
              {userFeedback && (
                <Typography variant="caption" color="text.secondary" textAlign="center" display="block">
                  Thank you for your feedback!
                </Typography>
              )}
            </Box>

            {/* Help Button */}
            <Box sx={{ textAlign: 'center', mt: 2 }}>
              <Button
                variant="outlined"
                startIcon={<HelpOutline />}
                onClick={onHelpClick}
                sx={{
                  borderColor: '#1976d2',
                  color: '#1976d2',
                  '&:hover': {
                    borderColor: '#9c27b0',
                    bgcolor: 'rgba(156, 39, 176, 0.05)',
                    color: '#9c27b0',
                  },
                  fontWeight: 600,
                  px: 3,
                  py: 1,
                }}
              >
                Why this result? Learn More
              </Button>
            </Box>
          </Stack>
        ) : (
          <Box sx={{ textAlign: 'center', py: 6 }}>
            <Avatar sx={{ bgcolor: 'grey.300', mx: 'auto', mb: 2, width: 64, height: 64 }}>
              <Psychology sx={{ fontSize: 32, color: 'grey.600' }} />
            </Avatar>
            <Typography variant="h6" color="text.secondary">
              No prediction yet
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Enter features and click predict
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
}
