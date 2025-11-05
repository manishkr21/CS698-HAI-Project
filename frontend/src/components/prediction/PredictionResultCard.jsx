import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Stack,
  Divider,
  Button,
  IconButton,
  Alert,
  Tooltip
} from '@mui/material';
import {
  ThumbUp,
  ThumbDown,
  Psychology,
  HelpOutline,
  Analytics
} from '@mui/icons-material';

export default function PredictionResultCard({ predictionResult, onHelpClick, onFeedback, onViewExplanation }) {
  const [feedback, setFeedback] = useState(null);
  const [showThankYou, setShowThankYou] = useState(false);

  const PREDICTION_CLASS = ["Enrolled", "Dropout", "Graduate"];

  const handleFeedback = (isPositive) => {
    if (feedback === isPositive) {
      setFeedback(null);
      onFeedback({ helpful: null });
    } else {
      setFeedback(isPositive);
      onFeedback({ helpful: isPositive });
      setShowThankYou(true);
      setTimeout(() => setShowThankYou(false), 3000);
    }
  };

  if (!predictionResult) {
    return (
      <Card>
        <CardContent>
          <Box sx={{ textAlign: 'center', py: 3 }}>
            <Psychology sx={{ fontSize: 40, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" color="text.secondary" gutterBottom>
              No Prediction Yet
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Enter features and click predict
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent>
        {/* Prediction Result Header */}
        <Box sx={{ textAlign: 'center', mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Prediction Result
          </Typography>
          <Typography 
            variant="h4" 
            color="primary" 
            sx={{ fontWeight: 700, mb: 1 }}
          >
            {PREDICTION_CLASS[predictionResult.prediction]}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {predictionResult.model_used}
          </Typography>
        </Box>

        {/* Class Probabilities */}
        {predictionResult.probabilities && (
          <Box sx={{ mb: 3, maxWidth: "400px", mx: 'auto' }} >
            <Typography variant="subtitle2" gutterBottom>
              Class Probabilities
            </Typography>
            <Stack spacing={2}>
              {Object.entries(predictionResult.probabilities)
                .sort((a, b) => b[1] - a[1]) // Sort by probability
                .map(([className, prob]) => (
                  <Box key={className}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                      <Typography variant="body2">
                        {className}
                      </Typography>
                      <Typography variant="body2" fontWeight="medium">
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

        {/* Action Buttons */}
        <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', mb: 3 }}>
          <Button
            variant="outlined"
            startIcon={<HelpOutline />}
            onClick={onHelpClick}
          >
            Help Me Understand
          </Button>
          <Button
            variant="outlined"
            startIcon={<Analytics />}
            onClick={onViewExplanation}
            color="secondary"
          >
            View Explanation
          </Button>
        </Box>

        {/* User Feedback Section */}
        <Box>
          <Divider sx={{ mb: 2 }} />
          <Typography variant="body2" color="text.secondary" textAlign="center" gutterBottom>
            Was this prediction helpful?
          </Typography>
          <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2 }}>
            <Tooltip title="This prediction was helpful">
              <IconButton
                onClick={() => handleFeedback(true)}
                color={feedback === true ? 'success' : 'default'}
              >
                <ThumbUp />
              </IconButton>
            </Tooltip>
            <Tooltip title="This prediction was not helpful">
              <IconButton
                onClick={() => handleFeedback(false)}
                color={feedback === false ? 'error' : 'default'}
              >
                <ThumbDown />
              </IconButton>
            </Tooltip>
          </Box>
          {showThankYou && (
            <Alert severity="success" sx={{ mt: 2 }}>
              Thank you for your feedback!
            </Alert>
          )}
        </Box>
      </CardContent>
    </Card>
  );
}
