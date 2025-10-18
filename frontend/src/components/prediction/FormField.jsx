import React from 'react';
import {
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Autocomplete,
  Box,
  Typography
} from '@mui/material';

export default function FormField({ feature, config, currentValue, predictionForm, setPredictionForm, getOptionsFromMapping }) {
  // Default to number field if no config
  if (!config) {
    return (
      <TextField
        fullWidth
        size="small"
        label={feature}
        type="number"
        value={currentValue ?? ''}
        onChange={(e) => setPredictionForm({
          ...predictionForm,
          [feature]: parseFloat(e.target.value) || 0
        })}
        variant="outlined"
        placeholder="Enter value"
      />
    );
  }

  // Autocomplete field
  if (config.type === 'autocomplete') {
    const options = getOptionsFromMapping(config.mapping);
    const selectedOption = options.find(opt => opt.value === currentValue) || null;

    return (
      <Autocomplete
        fullWidth
        size="small"
        options={options}
        value={selectedOption}
        getOptionLabel={(option) => option.label}
        sx={{ marginBottom: '2rem' }}
        onChange={(event, newValue) => {
          setPredictionForm({
            ...predictionForm,
            [feature]: newValue ? newValue.value : ''
          });
        }}
        renderInput={(params) => (
          <TextField 
            {...params} 
            label={feature} 
            variant="outlined"
            placeholder={`Select ${feature.toLowerCase()}`}
          />
        )}
        renderOption={(props, option) => (
          <li {...props} key={option.value}>
            <Box sx={{ display: 'flex', flexDirection: 'column' }}>
              <Typography variant="body2">{option.label}</Typography>
            </Box>
          </li>
        )}
        isOptionEqualToValue={(option, value) => option.value === value?.value}
      />
    );
  }

  // Select field
  if (config.type === 'select') {
    return (
      <FormControl fullWidth size="small">
        <InputLabel>{feature}</InputLabel>
        <Select
          value={currentValue ?? ''}
          label={feature}
          onChange={(e) => setPredictionForm({
            ...predictionForm,
            [feature]: e.target.value
          })}
          displayEmpty
        >
          {currentValue === '' || currentValue === undefined || currentValue === null ? (
            <MenuItem value="" disabled>
              <em>Select {feature.toLowerCase()}</em>
            </MenuItem>
          ) : null}
          {config.options.map((option) => (
            <MenuItem key={option.value} value={option.value}>
              {option.label}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
    );
  }

  // Number field with constraints
  if (config.type === 'number') {
    return (
      <TextField
        fullWidth
        size="small"
        label={feature}
        type="number"
        value={currentValue ?? ''}
        onChange={(e) => setPredictionForm({
          ...predictionForm,
          [feature]: parseFloat(e.target.value) || 0
        })}
        variant="outlined"
        placeholder="Enter value"
        inputProps={{
          min: config.min,
          max: config.max,
          step: config.step || 1
        }}
        helperText={config.min !== undefined && config.max !== undefined ? 
          `Range: ${config.min} - ${config.max}` : ''}
        sx={{ marginBottom: '2rem' }}
      />
    );
  }
}
