import React, { useState, useEffect } from 'react';
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  AppBar,
  Toolbar,
  Typography,
  Container,
  Box,
  Paper,
  Tabs,
  Tab,
  Button,
  Avatar,
  CircularProgress,
  useMediaQuery
} from '@mui/material';
import {
  CloudUpload,
  TrendingUp,
  Groups,
  Warning,
  Psychology,
  BarChart,
  Assessment,
  CompareArrows,
  HelpOutline,
  Analytics
} from '@mui/icons-material';

// Import modular components
import OverviewTab from './components/OverviewTab';
import PerformanceFairnessTab from './components/PerformanceFairnessTab';
import PredictionTab from './components/PredictionTab';
import CompareModelsTab from './components/CompareModelsTab';
import HelpTab from './components/HelpTab';
// import ExplanationTab from './components/old3543gdg/ExplanationTab';
import ModelExplanation from './components/ModelExplanation56';
import LocalExplanation from './components/LocalExplanation';
import logo from './assets/logo.png';
import config from './resources/config.json';
const API_URL = config.API_BASE_URL;

// Create Material-UI theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
      light: '#42a5f5',
      dark: '#1565c0',
    },
    secondary: {
      main: '#9c27b0',
      light: '#ba68c8',
      dark: '#7b1fa2',
    },
    success: {
      main: '#2e7d32',
      light: '#4caf50',
      dark: '#1b5e20',
    },
    warning: {
      main: '#ed6c02',
      light: '#ff9800',
      dark: '#e65100',
    },
    background: {
      default: '#f5f7fa',
      paper: '#ffffff',
    },
  },
  typography: {
    h4: {
      fontWeight: 700,
      background: 'linear-gradient(45deg, #1976d2 30%, #9c27b0 90%)',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent',
    },
    h5: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 600,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
          transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
          '&:hover': {
            transform: 'translateY(-4px)',
            boxShadow: '0 16px 48px rgba(0, 0, 0, 0.15)',
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          textTransform: 'none',
          fontWeight: 600,
          padding: '12px 24px',
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          fontSize: '1rem',
        },
      },
    },
  },
});

function TabPanel({ children, value, index, ...other }) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

export default function StudentDropoutDashboard() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [predictionForm, setPredictionForm] = useState({});
  const [predictionResult, setPredictionResult] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  const [error, setError] = useState(null);
  
  // New state for bias mitigation features
  const [biasMitigation, setBiasMitigation] = useState(null);
  const [comparisonResult, setComparisonResult] = useState(null);
  const [fairnessSummary, setFairnessSummary] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [predictionFeatures, setPredictionFeatures] = useState(null);
  const [selectedExplanationModel, setSelectedExplanationModel] = useState('best_model');

  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  useEffect(() => {
    fetchMetrics();
    fetchFeatures();
    fetchFairnessSummary();
  }, []);

  const fetchMetrics = async () => {
    try {
      const res = await fetch(`${API_URL}/metrics`);
      if (res.ok) {
        const data = await res.json();
        setMetrics(data);
        setError(null);
      } else {
        setError('No training data available. Please train models first.');
      }
    } catch (err) {
      setError('Backend not available. Make sure the server is running.');
    } finally {
      setLoading(false);
    }
  };

  const fetchFeatures = async () => {
    try {
      const res = await fetch(`${API_URL}/features`);
      const data = await res.json();
      
      // Initialize with realistic default values
      const defaultForm = {
        'Marital status': 1, // Single
        'Application mode': 1, // 1st phase - general contingent
        'Application order': 1,
        'Course': 9003, // Management
        'Daytime/evening attendance': 1, // Daytime
        'Previous qualification': 1, // Secondary education
        'Previous qualification (grade)': 120,
        'Nacionality': 1, // Portuguese
        "Mother's qualification": 1, // Secondary Education
        "Father's qualification": 1, // Secondary Education
        "Mother's occupation": 5, // Intermediate Level Technicians
        "Father's occupation": 5, // Intermediate Level Technicians
        'Admission grade': 120,
        'Displaced': null, // Not selected by default
        'Educational special needs': null, // Not selected by default
        'Debtor': null, // Not selected by default
        'Tuition fees up to date': null, // Not selected by default
        'Gender': 1, // Male
        'Scholarship holder': null, // Not selected by default
        'Age at enrollment': 20,
        'International': null, // Not selected by default
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
      
      setPredictionForm(defaultForm);
    } catch (err) {
      console.error('Failed to fetch features:', err);
    }
  };

  const fetchFairnessSummary = async () => {
    try {
      const res = await fetch(`${API_URL}/fairness/summary`);
      if (res.ok) {
        const data = await res.json();
        setFairnessSummary(data);
      }
    } catch (err) {
      console.log('Fairness summary not available:', err);
    }
  };

  const handlePredict = async () => {
    setPredicting(true);
    try {
      const features = Object.values(predictionForm);
      setPredictionFeatures(features); // Store features for explanations
      
      const res = await fetch(`${API_URL}/predict/with-bias-mitigation?bias_mitigation=${biasMitigation}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features })
      });
      
      const data = await res.json();
      setPredictionResult(data);
    } catch (err) {
      alert('Prediction failed: ' + err.message);
    } finally {
      setPredicting(false);
    }
  };

  const handleComparePredictions = async () => {
    setPredicting(true);
    try {
      const features = Object.values(predictionForm);
      const res = await fetch(`${API_URL}/predict/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features })
      });
      
      const data = await res.json();
      setComparisonResult(data);
    } catch (err) {
      alert('Comparison failed: ' + err.message);
    } finally {
      setPredicting(false);
    }
  };

  const handleNavigateToHelp = () => {
    setActiveTab(3); // Navigate to Help tab (index 3)
  };

  const handleNavigateToExplanation = () => {
    setActiveTab(2); // Navigate to Explanation tab
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      setLoading(true);
      const res = await fetch(`${API_URL}/train`, {
        method: 'POST',
        body: formData
      });

      if (res.ok) {
        alert('Models trained successfully!');
        await fetchMetrics();
      } else {
        const error = await res.json();
        alert('Training failed: ' + error.detail);
      }
    } catch (err) {
      alert('Upload failed: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Box
          sx={{
            minHeight: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'linear-gradient(135deg,rgb(89, 104, 172) 0%,rgb(72, 82, 218) 100%)',
          }}
        >
          <Paper elevation={8} sx={{ p: 6, borderRadius: 4, textAlign: 'center', maxWidth: 400 }}>
            <CircularProgress size={60} sx={{ mb: 3 }} />
            <Typography variant="h6" gutterBottom>Loading Dashboard...</Typography>
            <Typography variant="body2" color="text.secondary">Preparing your analytics</Typography>
          </Paper>
        </Box>
      </ThemeProvider>
    );
  }

  if (error) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Box
          sx={{
            minHeight: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'linear-gradient(135deg,rgb(89, 104, 172) 0%,rgb(72, 82, 218) 100%)',
            p: 3,
          }}
        >
          <Container maxWidth="md">
            <Paper elevation={8} sx={{ p: 4, textAlign: 'center', borderRadius: 4 }}>
              <Avatar sx={{ bgcolor: 'warning.main', width: 80, height: 80, mx: 'auto', mb: 3 }}>
                <Warning sx={{ fontSize: 40 }} />
              </Avatar>
              
              <Typography variant="h4" gutterBottom>No Training Data Available</Typography>
              <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>{error}</Typography>

              <Paper variant="outlined" sx={{ p: 4, bgcolor: 'grey.50', borderRadius: 3 }}>
                <Avatar sx={{ bgcolor: 'primary.main', width: 64, height: 64, mx: 'auto', mb: 2 }}>
                  <CloudUpload sx={{ fontSize: 32 }} />
                </Avatar>
                
                <Typography variant="h6" gutterBottom>Upload Training Data</Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                  Upload a CSV file with 36 features + Target column to start training your models
                </Typography>
                
                <Button
                  component="label"
                  variant="contained"
                  startIcon={<CloudUpload />}
                  size="large"
                  sx={{
                    background: 'linear-gradient(45deg, #1976d2 30%, #9c27b0 90%)',
                    boxShadow: '0 4px 20px rgba(25, 118, 210, 0.3)',
                  }}
                >
                  Choose CSV File
                  <input type="file" accept=".csv" onChange={handleFileUpload} hidden />
                </Button>
              </Paper>
            </Paper>
          </Container>
        </Box>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ bgcolor: 'background.default' }}>
        {/* App Bar */}
        <AppBar position="sticky" elevation={2}>
          <Toolbar sx={{ py: 1.5 }}>
            <img src={logo} alt="Logo" style={{ width: '60px', height: '60px', marginRight: '10px' }} />
            
            <Box sx={{ flexGrow: 1 }}>
              <Typography variant="h5" component="h1" sx={{ fontWeight: 700 }}>
                Student Dropout Analysis Platform
              </Typography>
              <Typography variant="body2" sx={{ opacity: 0.9, display: 'flex', alignItems: 'center', gap: 1 }}>
                Advanced ML Dashboard with Bias Mitigation & Fairness Analytics
              </Typography>
            </Box>

            <Button
              component="label"
              variant="contained"
              startIcon={<CloudUpload />}
              sx={{
                bgcolor: 'rgba(255, 255, 255, 0.2)',
                '&:hover': { bgcolor: 'rgba(255, 255, 255, 0.3)' },
                backdropFilter: 'blur(10px)',
              }}
            >
              Upload New Data
              <input type="file" accept=".csv" onChange={handleFileUpload} hidden />
            </Button>
          </Toolbar>
        </AppBar>

        <Container maxWidth="xl" sx={{ py: 3 }}>
          {/* Navigation Tabs */}
          <Paper elevation={2} sx={{ mb: 0, borderRadius: 3 }}>
            <Tabs
              value={activeTab}
              onChange={(e, newValue) => setActiveTab(newValue)}
              variant={isMobile ? "scrollable" : "fullWidth"}
              scrollButtons="auto"
              sx={{ px: 2 }}
            >
              <Tab icon={<Psychology />} label="Predict" iconPosition="start" />
              <Tab icon={<BarChart />} label="Performance & Fairness" iconPosition="start" />
              <Tab icon={<Analytics />} label="Explanations" iconPosition="start" />
              <Tab icon={<HelpOutline />} label="Help" iconPosition="start" />
            </Tabs>
          </Paper>

          {/* Tab Content */}
          <TabPanel value={activeTab} index={0}>
            <PredictionTab
              predictionForm={predictionForm}
              setPredictionForm={setPredictionForm}
              biasMitigation={biasMitigation}
              setBiasMitigation={setBiasMitigation}
              predictionResult={predictionResult}
              predicting={predicting}
              handlePredict={handlePredict}
              onNavigateToHelp={handleNavigateToHelp}
              onNavigateToExplanation={handleNavigateToExplanation}
            />
            {predictionResult && (
              <LocalExplanation
                selectedModel={selectedExplanationModel}
                features={predictionFeatures}
                featureNames={Object.keys(predictionForm)}
                prediction={predictionResult}
                classNames={['Dropout', 'Continue']}
              />
            )}
          </TabPanel>

          <TabPanel value={activeTab} index={1}>
            <PerformanceFairnessTab metrics={metrics} />
          </TabPanel>

          <TabPanel value={activeTab} index={2}>
            {/* <ExplanationTab 
              predictionResult={predictionResult}
              predictionFeatures={predictionFeatures}
            /> */}
            <ModelExplanation />
          </TabPanel>

          <TabPanel value={activeTab} index={3}>
            <HelpTab />
          </TabPanel>
        </Container>
      </Box>
    </ThemeProvider>
  );
}