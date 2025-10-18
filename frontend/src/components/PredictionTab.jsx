import React, { useState } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Grid,
  Typography,
  Avatar,
  Box,
  Button,
  Paper,
  CircularProgress,
  Divider,
  Tooltip,
  Stack
} from '@mui/material';
import { Psychology, Balance, Lightbulb } from '@mui/icons-material';

// Import mapping files
import maritalStatusMapping from '../mapping/marital_status.json';
import applicationModeMapping from '../mapping/application_mode.json';
import courseMapping from '../mapping/course.json';
import previousQualificationMapping from '../mapping/previous_qualification.json';
import nationalityMapping from '../mapping/nationality.json';
import motherQualificationMapping from '../mapping/mother_qualifications.json';
import fatherQualificationMapping from '../mapping/father_qualifications.json';
import motherOccupationMapping from '../mapping/mother_occupations.json';
import fatherOccupationMapping from '../mapping/father_occupations.json';

// Import new components
import BiasMethodSelector from './prediction/BiasMethodSelector';
import StudentInformationForm from './prediction/StudentInformationForm';
import ConsentStatusForm from './prediction/ConsentStatusForm';
import AcademicPerformanceForm from './prediction/AcademicPerformanceForm';
import EconomicIndicatorsForm from './prediction/EconomicIndicatorsForm';
import PredictionResultCard from './prediction/PredictionResultCard';

export default function PredictionTab({ 
  predictionForm, 
  setPredictionForm, 
  biasMitigation, 
  setBiasMitigation,
  predictionResult,
  predicting,
  handlePredict,
  onNavigateToHelp
}) {
  const [selectedCountry, setSelectedCountry] = useState(null);

  // Define features used by bias-mitigated model
  const biasMitigatedFeatures = [
    "Daytime/evening attendance",
    "Admission grade",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Scholarship holder",
    "Age at enrollment",
    "International",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)"
  ];

  // Economic indicators data
  const indiaEconomicIndicators = {
    'Unemployment rate': 7.5,
    'Inflation rate': 5.2,
    'GDP': 6.8
  };

  const portugalEconomicIndicators = {
    'Unemployment rate': 6.2,
    'Inflation rate': 2.8,
    'GDP': 2.3
  };

  // Example data
  const exampleData = {
    'Marital status': 1,
    'Application mode': 1,
    'Application order': 1,
    'Course': 9003,
    'Daytime/evening attendance': 1,
    'Previous qualification': 1,
    'Previous qualification (grade)': 120,
    'Nacionality': 1,
    "Mother's qualification": 1,
    "Father's qualification": 1,
    "Mother's occupation": 5,
    "Father's occupation": 5,
    'Admission grade': 120,
    'Displaced': 0,
    'Educational special needs': 0,
    'Debtor': 0,
    'Tuition fees up to date': 1,
    'Gender': 1,
    'Scholarship holder': 0,
    'Age at enrollment': 20,
    'International': 0,
    'Curricular units 1st sem (credited)': 0,
    'Curricular units 1st sem (enrolled)': 6,
    'Curricular units 1st sem (evaluations)': 6,
    'Curricular units 1st sem (approved)': 5,
    'Curricular units 1st sem (grade)': 13.5,
    'Curricular units 1st sem (without evaluations)': 0,
    'Curricular units 2nd sem (credited)': 0,
    'Curricular units 2nd sem (enrolled)': 6,
    'Curricular units 2nd sem (evaluations)': 6,
    'Curricular units 2nd sem (approved)': 5,
    'Curricular units 2nd sem (grade)': 13.0,
    'Curricular units 2nd sem (without evaluations)': 0,
    'Unemployment rate': 10.8,
    'Inflation rate': 1.4,
    'GDP': 1.74
  };

  // Field configuration
  const fieldConfig = {
    'Marital status': { type: 'autocomplete', mapping: maritalStatusMapping },
    'Application mode': { type: 'autocomplete', mapping: applicationModeMapping },
    'Application order': { type: 'number', min: 0, max: 9 },
    'Course': { type: 'autocomplete', mapping: courseMapping },
    'Daytime/evening attendance': { type: 'select', options: [{ value: 0, label: 'Evening' }, { value: 1, label: 'Daytime' }] },
    'Previous qualification': { type: 'autocomplete', mapping: previousQualificationMapping },
    'Previous qualification (grade)': { type: 'number', min: 0, max: 200, step: 0.1 },
    'Nacionality': { type: 'autocomplete', mapping: nationalityMapping },
    "Mother's qualification": { type: 'autocomplete', mapping: motherQualificationMapping },
    "Father's qualification": { type: 'autocomplete', mapping: fatherQualificationMapping },
    "Mother's occupation": { type: 'autocomplete', mapping: motherOccupationMapping },
    "Father's occupation": { type: 'autocomplete', mapping: fatherOccupationMapping },
    'Admission grade': { type: 'number', min: 0, max: 200, step: 0.1 },
    'Gender': { type: 'select', options: [{ value: 0, label: 'Female' }, { value: 1, label: 'Male' }] },
    'Age at enrollment': { type: 'number', min: 17, max: 70 },
  };

  // Field categories
  const consentFields = [
    'Displaced',
    'Educational special needs',
    'Debtor',
    'Tuition fees up to date',
    'Scholarship holder',
    'International'
  ];

  const semester1Fields = [
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)'
  ];

  const semester2Fields = [
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)'
  ];

  const economicFields = [
    'Unemployment rate',
    'Inflation rate',
    'GDP'
  ];

  // Filter fields based on selected model
  const getFilteredFields = (fields) => {
    if (biasMitigation === 'after') {
      return fields.filter(field => biasMitigatedFeatures.includes(field));
    }
    return fields;
  };

  const filteredConsentFields = getFilteredFields(consentFields);
  const filteredSemester1Fields = getFilteredFields(semester1Fields);
  const filteredSemester2Fields = getFilteredFields(semester2Fields);

  const generalFields = Object.keys(predictionForm).filter(
    field => !semester1Fields.includes(field) && 
             !semester2Fields.includes(field) && 
             !economicFields.includes(field) &&
             !consentFields.includes(field)
  );

  const filteredGeneralFields = getFilteredFields(generalFields);

  // Utility functions
  const getOptionsFromMapping = (mapping) => {
    return Object.entries(mapping).map(([value, label]) => ({
      value: parseInt(value),
      label: label
    }));
  };

  const handleTryExample = () => {
    setPredictionForm(exampleData);
    setSelectedCountry(null);
  };

  const handleIndiaFlagClick = () => {
    setPredictionForm({
      ...predictionForm,
      ...indiaEconomicIndicators
    });
    setSelectedCountry('India');
  };

  const handlePortugalFlagClick = () => {
    setPredictionForm({
      ...predictionForm,
      ...portugalEconomicIndicators
    });
    setSelectedCountry('Portugal');
  };

  return (
    <Box>
      {/* Header with Bias Method Toggle */}
      <Card sx={{ mb: 3, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white' }}>
        <CardContent>
          <Stack direction="row" justifyContent="space-between" alignItems="center" flexWrap="wrap" gap={2}>
            <Box>
              <Typography variant="h5" fontWeight={700} gutterBottom>
                Make Prediction
              </Typography>
              <Typography variant="body2" sx={{ opacity: 0.9 }}>
                {biasMitigation === 'after' 
                  ? 'Bias-Mitigated Model uses only fair features (excluding sensitive attributes)'
                  : 'Select bias mitigation approach and enter student features'}
              </Typography>
            </Box>
            
            {/* Bias Method Selector - Integrated in Header */}
            <BiasMethodSelector 
              biasMitigation={biasMitigation}
              onBiasChange={setBiasMitigation}
            />
          </Stack>
        </CardContent>
      </Card>

      {/* Show form only if bias mitigation is selected */}
      {biasMitigation ? (
        <>
          {/* Form Card */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              {/* Try with Example Button */}
              <Box sx={{ mb: 3, textAlign: 'center' }}>
                <Tooltip title="Auto-fill all fields with example student data" arrow>
                  <Button
                    variant="contained"
                    size="large"
                    startIcon={<Lightbulb />}
                    onClick={handleTryExample}
                    sx={{
                      bgcolor: '#FFA726',
                      color: '#fff',
                      '&:hover': {
                        bgcolor: '#FF9800',
                        transform: 'scale(1.05)',
                      },
                      transition: 'all 0.3s',
                      fontWeight: 600,
                      boxShadow: '0 4px 16px rgba(255, 167, 38, 0.4)',
                      px: 4,
                      py: 1.5,
                    }}
                  >
                    Try with Example
                  </Button>
                </Tooltip>
              </Box>

              {/* Form Fields Container */}
              <Paper variant="outlined" sx={{ p: 3, bgcolor: 'grey.50', maxHeight: 600, overflow: 'auto' }}>
                {/* Student Information - Only show if there are filtered fields */}
                {filteredGeneralFields.length > 0 && (
                  <>
                    <StudentInformationForm
                      generalFields={filteredGeneralFields}
                      fieldConfig={fieldConfig}
                      predictionForm={predictionForm}
                      setPredictionForm={setPredictionForm}
                      getOptionsFromMapping={getOptionsFromMapping}
                    />
                    <Divider sx={{ my: 3 }} />
                  </>
                )}

                {/* Consent & Status */}
                {filteredConsentFields.length > 0 && (
                  <>
                    <ConsentStatusForm
                      consentFields={filteredConsentFields}
                      predictionForm={predictionForm}
                      setPredictionForm={setPredictionForm}
                    />
                    <Divider sx={{ my: 3 }} />
                  </>
                )}

                {/* Academic Performance */}
                {(filteredSemester1Fields.length > 0 || filteredSemester2Fields.length > 0) && (
                  <>
                    <AcademicPerformanceForm
                      semester1Fields={filteredSemester1Fields}
                      semester2Fields={filteredSemester2Fields}
                      fieldConfig={fieldConfig}
                      predictionForm={predictionForm}
                      setPredictionForm={setPredictionForm}
                      getOptionsFromMapping={getOptionsFromMapping}
                    />
                    <Divider sx={{ my: 3 }} />
                  </>
                )}

                {/* Economic Indicators - Only show for baseline model */}
                {biasMitigation === 'baseline' && (
                  <EconomicIndicatorsForm
                    predictionForm={predictionForm}
                    setPredictionForm={setPredictionForm}
                    selectedCountry={selectedCountry}
                    onIndiaClick={handleIndiaFlagClick}
                    onPortugalClick={handlePortugalFlagClick}
                  />
                )}
              </Paper>
              
              {/* Predict Button */}
              <Box sx={{ mt: 3 }}>
                <Button
                  fullWidth
                  variant="contained"
                  size="large"
                  onClick={handlePredict}
                  startIcon={predicting ? <CircularProgress size={20} color="inherit" /> : <Psychology />}
                  disabled={predicting}
                  sx={{
                    background: 'linear-gradient(45deg, #9c27b0 30%, #1976d2 90%)',
                    py: 2,
                  }}
                >
                  {predicting ? 'Generating Prediction...' : 'Generate Prediction'}
                </Button>
              </Box>
            </CardContent>
          </Card>

          {/* Prediction Result - Show at Bottom Only After Prediction */}
          {predictionResult && (
            <PredictionResultCard 
              predictionResult={predictionResult}
              onHelpClick={onNavigateToHelp}
            />
          )}
        </>
      ) : (
        <Card>
          <CardContent>
            <Box sx={{ textAlign: 'center', py: 8 }}>
              <Avatar sx={{ bgcolor: 'primary.50', mx: 'auto', mb: 3, width: 80, height: 80 }}>
                <Balance sx={{ fontSize: 40, color: 'primary.main' }} />
              </Avatar>
              <Typography variant="h6" color="text.secondary" gutterBottom>
                Please Select a Student Dropout Prediction Model
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Choose between Baseline Model or Bias-Mitigated Model to continue
              </Typography>
            </Box>
          </CardContent>
        </Card>
      )}
    </Box>
  );
}
