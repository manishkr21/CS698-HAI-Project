import React from 'react';
import { Box, Typography, Grid, Paper, Stack } from '@mui/material';
import { CalendarToday } from '@mui/icons-material';
import FormField from './FormField';

export default function AcademicPerformanceForm({ 
  semester1Fields, 
  semester2Fields, 
  fieldConfig, 
  predictionForm, 
  setPredictionForm, 
  getOptionsFromMapping 
}) {
  return (
    <Box sx={{ mb: 4 }}>
      <Typography variant="h6" gutterBottom sx={{ color: 'primary.main', fontWeight: 600, mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
        <CalendarToday fontSize="small" />
        Academic Performance
      </Typography>
      
      <Grid container spacing={3}>
        {/* Semester 1 - Left Side */}
        <Grid item xs={12} md={6} width={"35%"}>
          <Paper elevation={2} sx={{ p: 2.5, bgcolor: 'info.50', borderLeft: '4px solid', borderColor: 'info.main' }}>
            <Typography variant="subtitle1" fontWeight="600" gutterBottom sx={{ color: 'info.main', mb: 2 }}>
              First Semester
            </Typography>
            <Stack spacing={2}>
              {semester1Fields.map(feature => (
                <Box key={feature}>
                  <FormField
                    feature={feature}
                    config={fieldConfig[feature]}
                    currentValue={predictionForm[feature]}
                    predictionForm={predictionForm}
                    setPredictionForm={setPredictionForm}
                    getOptionsFromMapping={getOptionsFromMapping}
                  />
                </Box>
              ))}
            </Stack>
          </Paper>
        </Grid>

        {/* Semester 2 - Right Side */}
        <Grid item xs={12} md={6} width={"35%"}>
          <Paper elevation={2} sx={{ p: 2.5, bgcolor: 'success.50', borderLeft: '4px solid', borderColor: 'success.main' }}>
            <Typography variant="subtitle1" fontWeight="600" gutterBottom sx={{ color: 'success.main', mb: 2 }}>
              Second Semester
            </Typography>
            <Stack spacing={2}>
              {semester2Fields.map(feature => (
                <Box key={feature}>
                  <FormField
                    feature={feature}
                    config={fieldConfig[feature]}
                    currentValue={predictionForm[feature]}
                    predictionForm={predictionForm}
                    setPredictionForm={setPredictionForm}
                    getOptionsFromMapping={getOptionsFromMapping}
                  />
                </Box>
              ))}
            </Stack>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}
