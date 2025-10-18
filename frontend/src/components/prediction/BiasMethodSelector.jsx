import React from 'react';
import { Box, ToggleButtonGroup, ToggleButton } from '@mui/material';
import { BarChart, Balance } from '@mui/icons-material';

export default function BiasMethodSelector({ biasMitigation, onBiasChange }) {
  const handleBiasToggle = (event, newValue) => {
    // Allow null value to be set (deselect)
    onBiasChange(newValue);
  };

  return (
    <Box sx={{ display: 'flex', justifyContent: 'center' }}>
      <ToggleButtonGroup
        value={biasMitigation}
        exclusive
        onChange={handleBiasToggle}
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
        <ToggleButton value="baseline">
          <BarChart sx={{ mr: 1 }} fontSize="small" />
          Baseline Model
        </ToggleButton>
        <ToggleButton value="after">
          <Balance sx={{ mr: 1 }} fontSize="small" />
          Bias-Mitigated Model
        </ToggleButton>
      </ToggleButtonGroup>
    </Box>
  );
}
