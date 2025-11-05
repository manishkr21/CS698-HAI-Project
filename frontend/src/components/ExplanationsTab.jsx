import React, { useState, useEffect } from 'react';
import {
  Card,
  CardHeader,
  CardContent,
  Typography,
  Box,
  Grid,
  Avatar,
  List,
  ListItem,
  ListItemText,
  Divider,
  Alert,
  CircularProgress
} from '@mui/material';
import { BarChart, Insights, Assessment, ShowChart } from '@mui/icons-material';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function ExplanationsTab({ predictionForm }) {
  const [loading, setLoading] = useState(false);
  const [globalImportance, setGlobalImportance] = useState(null);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    fetchGlobalImportance();
  }, []);
  
  const fetchGlobalImportance = async () => {
    try {
      setLoading(true);
      const res = await fetch(`${API_URL}/explain/global/feature-importance`);
      const data = await res.json();
      setGlobalImportance(data);
      setError(null);
    } catch (err) {
      setError("Failed to fetch feature importance: " + err.message);
    } finally {
      setLoading(false);
    }
  };
  
  const getPlotUrl = (type, params = '') => {
    if (!predictionForm || Object.keys(predictionForm).length === 0) return null;
    
    const features = Object.values(predictionForm);
    const queryParams = `features=${features.join(',')}`;
    
    return `${API_URL}/explain/${type}?${queryParams}`;
  };
  
  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
      </Box>
    );
  }
  
  if (error) {
    return (
      <Alert severity="error" sx={{ mt: 2 }}>
        {error}
      </Alert>
    );
  }
  
  return (
    <Box>
      <Grid container spacing={3}>
        {/* Global Feature Importance */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader
              avatar={<Avatar sx={{ bgcolor: 'primary.main' }}><BarChart /></Avatar>}
              title="Global Feature Importance"
              titleTypographyProps={{ variant: 'h6' }}
            />
            <CardContent>
              {globalImportance && (
                <List>
                  {globalImportance.top_features.map((feature, index) => (
                    <React.Fragment key={feature}>
                      <ListItem>
                        <ListItemText
                          primary={feature}
                          secondary={`Impact Score: ${globalImportance.feature_importance[feature].toFixed(4)}`}
                        />
                      </ListItem>
                      {index < globalImportance.top_features.length - 1 && <Divider />}
                    </React.Fragment>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>
        
        {/* Global Summary Plot */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader
              avatar={<Avatar sx={{ bgcolor: 'success.main' }}><Assessment /></Avatar>}
              title="SHAP Summary Plot"
              titleTypographyProps={{ variant: 'h6' }}
            />
            <CardContent>
              <Box sx={{ textAlign: 'center' }}>
                <img 
                  src={`${API_URL}/explain/global/summary-plot`}
                  alt="SHAP Summary Plot"
                  style={{ maxWidth: '100%', height: 'auto' }}
                  loading="lazy"
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Local Explanations */}
        {predictionForm && Object.keys(predictionForm).length > 0 && (
          <>
            <Grid item xs={12}>
              <Card>
                <CardHeader
                  avatar={<Avatar sx={{ bgcolor: 'info.main' }}><Insights /></Avatar>}
                  title="Feature Contributions (Waterfall Plot)"
                  titleTypographyProps={{ variant: 'h6' }}
                />
                <CardContent>
                  <Box sx={{ textAlign: 'center' }}>
                    <img 
                      src={getPlotUrl('local/waterfall/1')}  // Show for the second class (usually most interesting)
                      alt="SHAP Waterfall Plot"
                      style={{ maxWidth: '100%', height: 'auto' }}
                      loading="lazy"
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12}>
              <Card>
                <CardHeader
                  avatar={<Avatar sx={{ bgcolor: 'warning.main' }}><ShowChart /></Avatar>}
                  title="Feature Impact Flow"
                  titleTypographyProps={{ variant: 'h6' }}
                />
                <CardContent>
                  <Box sx={{ textAlign: 'center' }}>
                    <img 
                      src={getPlotUrl('local/force-plot')}
                      alt="SHAP Force Plot"
                      style={{ maxWidth: '100%', height: 'auto' }}
                      loading="lazy"
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </>
        )}
      </Grid>
    </Box>
  );
}