import React from 'react';
import { Box, Typography, Grid } from '@mui/material';
import FormField from './FormField';

export default function StudentInformationForm({ generalFields, fieldConfig, predictionForm, setPredictionForm, getOptionsFromMapping }) {
  return (
    <Box sx={{ mb: 4 }}>
      <Typography variant="h6" gutterBottom sx={{ color: 'primary.main', fontWeight: 600, mb: 2 }}>
        Student Information
      </Typography>
      <Grid container spacing={6}>
        {generalFields.map(feature => (
          <Grid item xs={12} sm={3} md={3} key={feature} width={"30%"}>
            <FormField
              feature={feature}
              config={fieldConfig[feature]}
              currentValue={predictionForm[feature]}
              predictionForm={predictionForm}
              setPredictionForm={setPredictionForm}
              getOptionsFromMapping={getOptionsFromMapping}
            />
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}
