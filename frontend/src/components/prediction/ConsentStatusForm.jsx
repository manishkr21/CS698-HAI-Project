import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  FormControl,
  FormLabel,
  RadioGroup,
  FormControlLabel,
  Radio
} from '@mui/material';

export default function ConsentStatusForm({ consentFields, predictionForm, setPredictionForm }) {
  return (
    <Box sx={{ mb: 4 }}>
      <Typography variant="h6" gutterBottom sx={{ color: 'primary.main', fontWeight: 600, mb: 2 }}>
        Consent & Status
      </Typography>
      <Paper elevation={2} sx={{ p: 3, bgcolor: 'warning.50', borderLeft: '4px solid', borderColor: 'warning.main' }}>
        <Grid container spacing={12}>
          {consentFields.map(field => (
            <Grid item xs={12} sm={6} md={4} key={field}>
              <FormControl component="fieldset">
                <FormLabel component="legend" sx={{ fontSize: '0.875rem', fontWeight: 500, mb: 1 }}>
                  {field}
                </FormLabel>
                <RadioGroup
                  row
                  value={predictionForm[field] !== undefined && predictionForm[field] !== null && predictionForm[field] !== '' ? predictionForm[field] : ''}
                  onChange={(e) => setPredictionForm({
                    ...predictionForm,
                    [field]: parseInt(e.target.value)
                  })}
                >
                  <FormControlLabel 
                    value={1} 
                    control={<Radio size="small" />} 
                    label="Yes" 
                    sx={{ mr: 3 }}
                  />
                  <FormControlLabel 
                    value={0} 
                    control={<Radio size="small" />} 
                    label="No" 
                  />
                </RadioGroup>
              </FormControl>
            </Grid>
          ))}
        </Grid>
      </Paper>
    </Box>
  );
}
