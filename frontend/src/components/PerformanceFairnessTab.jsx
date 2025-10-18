import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  ToggleButton,
  ToggleButtonGroup,
  Paper,
  Stack,
  Divider
} from '@mui/material';
import {
  BarChart,
  Groups,
  TrendingUp,
  Security
} from '@mui/icons-material';
import PerformanceTab from './PerformanceTab';
import FairnessTab from './FairnessTab';

export default function PerformanceFairnessTab({ metrics }) {
  const [view, setView] = useState('performance');

  return (
    <Box>
      {/* Header with View Toggle */}
      <Card sx={{ mb: 3, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white' }}>
        <CardContent>
          <Stack direction="row" justifyContent="space-between" alignItems="center" flexWrap="wrap" gap={2}>
            <Box>
              <Typography variant="h5" fontWeight={700} gutterBottom>
                Performance & Fairness Analytics
              </Typography>
              <Typography variant="body2" sx={{ opacity: 0.9 }}>
                Comprehensive model evaluation and fairness metrics
              </Typography>
            </Box>
            
            <ToggleButtonGroup
              value={view}
              exclusive
              onChange={(e, newView) => newView && setView(newView)}
              sx={{
                bgcolor: 'rgba(255, 255, 255, 0.15)',
                backdropFilter: 'blur(10px)',
                '& .MuiToggleButton-root': {
                  color: 'white',
                  borderColor: 'rgba(255, 255, 255, 0.3)',
                  '&.Mui-selected': {
                    bgcolor: 'rgba(239, 225, 250, 0.9)',
                    color: '#764ba2',
                    '&:hover': {
                      bgcolor: 'rgba(207, 169, 236, 0.9)',
                    }
                  },
                  '&:hover': {
                    bgcolor: 'rgba(255, 255, 255, 0.1)',
                  }
                }
              }}
            >
              <ToggleButton value="performance">
                <BarChart sx={{ mr: 1 }} />
                Performance
              </ToggleButton>
              <ToggleButton value="fairness">
                <Groups sx={{ mr: 1 }} />
                Fairness
              </ToggleButton>
            </ToggleButtonGroup>
          </Stack>
        </CardContent>
      </Card>

      {/* Content */}
      {view === 'performance' ? (
        <PerformanceTab metrics={metrics} />
      ) : (
        <FairnessTab metrics={metrics} />
      )}
    </Box>
  );
}
