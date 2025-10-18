import React from 'react';
import { Box, Typography, Stack, Slider, Button, Tooltip } from '@mui/material';

export default function EconomicIndicatorsForm({ 
  predictionForm, 
  setPredictionForm, 
  selectedCountry,
  onIndiaClick,
  onPortugalClick
}) {
  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box>
          <Typography variant="h6" sx={{ color: 'primary.main', fontWeight: 600 }}>
            Economic Indicators
          </Typography>
          {selectedCountry && (
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5 }}>
              {selectedCountry === 'India' ? '🇮🇳' : '🇵🇹'} Currently showing {selectedCountry}'s economic indicators
            </Typography>
          )}
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Tooltip title="Auto-fill with India's current economic indicators" arrow>
            <Button
              variant={selectedCountry === 'India' ? 'contained' : 'outlined'}
              size="small"
              onClick={onIndiaClick}
              sx={{
                borderColor: '#FF9933',
                color: selectedCountry === 'India' ? '#fff' : '#FF9933',
                bgcolor: selectedCountry === 'India' ? '#FF9933' : 'transparent',
                '&:hover': {
                  borderColor: '#138808',
                  bgcolor: selectedCountry === 'India' ? '#FF9933' : 'rgba(255, 153, 51, 0.1)',
                },
                transition: 'all 0.3s',
                fontWeight: selectedCountry === 'India' ? 600 : 400,
              }}
            >
              🇮🇳 India
            </Button>
          </Tooltip>
          <Tooltip title="Auto-fill with Portugal's current economic indicators" arrow>
            <Button
              variant={selectedCountry === 'Portugal' ? 'contained' : 'outlined'}
              size="small"
              onClick={onPortugalClick}
              sx={{
                borderColor: '#006600',
                color: selectedCountry === 'Portugal' ? '#fff' : '#006600',
                bgcolor: selectedCountry === 'Portugal' ? '#006600' : 'transparent',
                '&:hover': {
                  borderColor: '#FF0000',
                  bgcolor: selectedCountry === 'Portugal' ? '#006600' : 'rgba(0, 102, 0, 0.1)',
                },
                transition: 'all 0.3s',
                fontWeight: selectedCountry === 'Portugal' ? 600 : 400,
              }}
            >
              🇵🇹 Portugal
            </Button>
          </Tooltip>
        </Box>
      </Box>
      <Stack spacing={3}>
        {/* Unemployment Rate Slider */}
        <Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="body2" fontWeight="medium">
              Unemployment Rate = {predictionForm['Unemployment rate']?.toFixed(1) || 0}%
            </Typography>
          </Box>
          <Slider
            value={predictionForm['Unemployment rate'] || 0}
            onChange={(e, newValue) => setPredictionForm({
              ...predictionForm,
              'Unemployment rate': newValue
            })}
            min={0}
            max={20}
            step={0.1}
            valueLabelDisplay="auto"
            valueLabelFormat={(value) => `${value}%`}
            marks={[
              { value: 0, label: '0%' },
              { value: 5, label: '5%' },
              { value: 10, label: '10%' },
              { value: 15, label: '15%' },
              { value: 20, label: '20%' }
            ]}
            sx={{
              '& .MuiSlider-thumb': { bgcolor: '#FF9933' },
              '& .MuiSlider-track': { bgcolor: '#FF9933' },
              '& .MuiSlider-rail': { bgcolor: 'grey.300' },
            }}
          />
        </Box>

        {/* Inflation Rate Slider */}
        <Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="body2" fontWeight="medium">
              Inflation Rate = {predictionForm['Inflation rate']?.toFixed(1) || 0}%
            </Typography>
          </Box>
          <Slider
            value={predictionForm['Inflation rate'] || 0}
            onChange={(e, newValue) => setPredictionForm({
              ...predictionForm,
              'Inflation rate': newValue
            })}
            min={-5}
            max={15}
            step={0.1}
            valueLabelDisplay="auto"
            valueLabelFormat={(value) => `${value}%`}
            marks={[
              { value: -5, label: '-5%' },
              { value: 0, label: '0%' },
              { value: 5, label: '5%' },
              { value: 10, label: '10%' },
              { value: 15, label: '15%' }
            ]}
            sx={{
              '& .MuiSlider-thumb': { bgcolor: '#FFFFFF', border: '2px solid #138808' },
              '& .MuiSlider-track': { bgcolor: '#138808' },
              '& .MuiSlider-rail': { bgcolor: 'grey.300' },
            }}
          />
        </Box>

        {/* GDP Slider */}
        <Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="body2" fontWeight="medium">
              GDP Growth Rate = {predictionForm['GDP']?.toFixed(1) || 0}%
            </Typography>
          </Box>
          <Slider
            value={predictionForm['GDP'] || 0}
            onChange={(e, newValue) => setPredictionForm({
              ...predictionForm,
              'GDP': newValue
            })}
            min={-10}
            max={15}
            step={0.1}
            valueLabelDisplay="auto"
            valueLabelFormat={(value) => `${value}%`}
            marks={[
              { value: -10, label: '-10%' },
              { value: -5, label: '-5%' },
              { value: 0, label: '0%' },
              { value: 5, label: '5%' },
              { value: 10, label: '10%' },
              { value: 15, label: '15%' }
            ]}
            sx={{
              '& .MuiSlider-thumb': { bgcolor: '#000080' },
              '& .MuiSlider-track': { bgcolor: '#000080' },
              '& .MuiSlider-rail': { bgcolor: 'grey.300' },
            }}
          />
        </Box>
      </Stack>
    </Box>
  );
}
