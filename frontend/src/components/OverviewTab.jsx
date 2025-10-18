import React from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Grid,
  Typography,
  Avatar,
  Stack
} from '@mui/material';
import {
  TrendingUp,
  EmojiEvents,
  BarChart as BarChartIcon
} from '@mui/icons-material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function OverviewTab({ metrics }) {
  const prepareOverviewData = () => {
    if (!metrics || !metrics.results) return [];
    
    return Object.entries(metrics.results)
      .filter(([name]) => name !== 'mitigated')
      .map(([name, data]) => ({
        name: name.replace(/_/g, ' '),
        accuracy: (data.test_accuracy * 100).toFixed(2)
      }));
  };

  const overviewData = prepareOverviewData();

  return (
    <Grid container spacing={4}>
      <Grid item xs={12} lg={8} width={"78%"}>
        <Card>
          <CardHeader
            avatar={<Avatar sx={{ bgcolor: 'primary.main' }}><TrendingUp /></Avatar>}
            title="Model Comparison"
            titleTypographyProps={{ variant: 'h6' }}
          />
          <CardContent>
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={overviewData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1976d2', 
                    border: 'none', 
                    borderRadius: '8px',
                    color: 'white'
                  }} 
                />
                <Bar dataKey="accuracy" fill="url(#colorGradient)" radius={[4, 4, 0, 0]} />
                <defs>
                  <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#1976d2" />
                    <stop offset="100%" stopColor="#9c27b0" />
                  </linearGradient>
                </defs>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} lg={4} width={"19%"} >
        <Card>
          <CardHeader
            avatar={<Avatar sx={{ bgcolor: 'success.main' }}><EmojiEvents /></Avatar>}
            title="Training Summary"
            titleTypographyProps={{ variant: 'h6' }}
          />
          <CardContent>
            <Stack spacing={3}>
              <Card variant="outlined" sx={{ p: 2, bgcolor: 'primary.50' }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Best Model
                </Typography>
                <Typography variant="h6" color="primary.main">
                  {metrics?.best_model?.replace(/_/g, ' ')}
                </Typography>
              </Card>
              
              <Card variant="outlined" sx={{ p: 2, bgcolor: 'success.50' }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Models Trained
                </Typography>
                <Typography variant="h6" color="success.main">
                  {metrics?.models.length}
                </Typography>
              </Card>
              
              <Card variant="outlined" sx={{ p: 2, bgcolor: 'info.50' }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Training Time
                </Typography>
                <Typography variant="h6" color="info.main" sx={{paddingBottom: 3}}>
                  {metrics?.training_time_seconds.toFixed(1)}s
                </Typography>
              </Card>
            </Stack>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
}
