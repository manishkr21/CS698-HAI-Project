import React from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Grid,
  Typography,
  Avatar,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Box,
  Divider,
  Alert,
  Stack
} from '@mui/material';
import { 
  BarChart as BarChartIcon, 
  Assessment, 
  TrendingUp,
  CheckCircle,
  EmojiEvents
} from '@mui/icons-material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function PerformanceTab({ metrics }) {
  // Cross-validation comparison data
  const crossValidationData = [
    {
      metric: 'Validation Accuracy',
      'Random Forest': 0.7976,
      'Gradient Boosting': 0.8095,
      'Logistic Regression': 0.7250
    },
    {
      metric: 'Test Accuracy',
      'Random Forest': 0.7462,
      'Gradient Boosting': 0.7521,
      'Logistic Regression': 0.7238
    },
    {
      metric: 'Precision',
      'Random Forest': 0.8051,
      'Gradient Boosting': 0.8124,
      'Logistic Regression': 0.7267
    },
    {
      metric: 'Recall',
      'Random Forest': 0.7976,
      'Gradient Boosting': 0.8095,
      'Logistic Regression': 0.7250
    },
    {
      metric: 'F1-Score',
      'Random Forest': 0.7978,
      'Gradient Boosting': 0.8090,
      'Logistic Regression': 0.7240
    }
  ];

  const modelComparisonTable = [
    {
      model: 'Random Forest',
      validationAcc: '79.76%',
      testAcc: '74.62%',
      precision: '80.51%',
      recall: '79.76%',
      f1: '79.78%',
      rank: 2
    },
    {
      model: 'Gradient Boosting',
      validationAcc: '80.95%',
      testAcc: '75.21%',
      precision: '81.24%',
      recall: '80.95%',
      f1: '80.90%',
      rank: 1
    },
    {
      model: 'Logistic Regression',
      validationAcc: '72.50%',
      testAcc: '72.38%',
      precision: '72.67%',
      recall: '72.50%',
      f1: '72.40%',
      rank: 3
    }
  ];

  const prepareClassPerformance = () => {
    if (!metrics || !metrics.results || !metrics.best_model) return [];
    
    const bestModel = metrics.results[metrics.best_model];
    const report = bestModel.classification_report;
    
    return metrics.class_names.map((className) => ({
      class: className,
      precision: (report[className]?.precision * 100 || 0).toFixed(1),
      recall: (report[className]?.recall * 100 || 0).toFixed(1),
      f1: (report[className]?.['f1-score'] * 100 || 0).toFixed(1)
    }));
  };

  const classData = prepareClassPerformance();

  return (
    <Box>
      {/* Cross-Validation Results Comparison */}
      <Card sx={{ mb: 4 }}>
        <CardHeader
          // avatar={
          //   <Avatar sx={{ bgcolor: 'primary.main', width: 56, height: 56 }}>
          //     <BarChartIcon sx={{ fontSize: 32 }} />
          //   </Avatar>
          // }
          title={
            <Typography variant="h5" fontWeight={700} textAlign={'center'}>
              Student Dropout Prediction Models Performace - Bar Chart
            </Typography>
          }
         
        />
        <CardContent>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={crossValidationData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis 
                dataKey="metric" 
                tick={{ fontSize: 13, fontWeight: 600 }} 
                angle={-15}
                textAnchor="end"
                height={80}
              />
              <YAxis 
                domain={[0, 1]} 
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
              />
              <Tooltip 
                formatter={(value) => `${(value * 100).toFixed(2)}%`}
                contentStyle={{ 
                  backgroundColor: 'rgba(255, 255, 255, 0.95)', 
                  border: '2px solid #1976d2', 
                  borderRadius: '12px',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
                }} 
              />
              <Legend 
                wrapperStyle={{ paddingTop: '20px' }}
                iconType="rect"
              />
              <Bar dataKey="Random Forest" fill="#1976d2" radius={[8, 8, 0, 0]} />
              <Bar dataKey="Gradient Boosting" fill="#ff9800" radius={[8, 8, 0, 0]} />
              <Bar dataKey="Logistic Regression" fill="#4caf50" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Model Comparison Table */}
      <Card sx={{ mb: 4 }}>
        <CardHeader
          // avatar={
          //   <Avatar sx={{ bgcolor: 'secondary.main', width: 56, height: 56 }}>
          //     <Assessment sx={{ fontSize: 32 }} />
          //   </Avatar>
          // }
          title={
            <Typography variant="h5" fontWeight={700}>
              Student Dropout Prediction Models Performace - Table
            </Typography>
          }
        />
        <CardContent>
          <TableContainer component={Paper} variant="outlined" sx={{ borderRadius: 2 }}>
            <Table>
              <TableHead sx={{ bgcolor: 'grey.100' }}>
                <TableRow>
                  <TableCell><strong>Model</strong></TableCell>
                  <TableCell align="center"><strong>Validation Accuracy</strong></TableCell>
                  <TableCell align="center"><strong>Test Accuracy</strong></TableCell>
                  <TableCell align="center"><strong>Precision</strong></TableCell>
                  <TableCell align="center"><strong>Recall</strong></TableCell>
                  <TableCell align="center"><strong>F1-Score</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {modelComparisonTable.map((row, idx) => (
                  <TableRow 
                    key={idx} 
                    hover
                    sx={{ 
                      bgcolor: row.rank === 1 ? 'success.lighter' : 'inherit',
                      '&:hover': { bgcolor: row.rank === 1 ? 'success.light' : 'action.hover' }
                    }}
                  >
                    <TableCell>
                      <Stack direction="row" alignItems="center" spacing={1}>
                        {row.rank === 1 && <EmojiEvents color="warning" />}
                        <Typography fontWeight={row.rank === 1 ? 700 : 600}>
                          {row.model}
                        </Typography>
                        {row.rank === 1 && (
                          <Chip label="Best" color="success" size="small" />
                        )}
                      </Stack>
                    </TableCell>
                    <TableCell align="center">
                      <Chip 
                        label={row.validationAcc} 
                        // color={row.rank === 1 ? "success" : "default"} 
                        variant={row.rank === 1 ? "filled" : "outlined"}
                      />
                    </TableCell>
                    <TableCell align="center">
                      <Chip 
                        label={row.testAcc} 
                        // color={row.rank === 1 ? "success" : "default"} 
                        variant={row.rank === 1 ? "filled" : "outlined"}
                      />
                    </TableCell>
                    <TableCell align="center">
                      <Chip 
                        label={row.precision} 
                        // color={row.rank === 1 ? "success" : "default"} 
                        variant={row.rank === 1 ? "filled" : "outlined"}
                      />
                    </TableCell>
                    <TableCell align="center">
                      <Chip 
                        label={row.recall} 
                        // color={row.rank === 1 ? "success" : "default"} 
                        variant={row.rank === 1 ? "filled" : "outlined"}
                      />
                    </TableCell>
                    <TableCell align="center">
                      <Chip 
                        label={row.f1} 
                        // color={row.rank === 1 ? "success" : "default"} 
                        variant={row.rank === 1 ? "filled" : "outlined"}
                      />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {/* Key Insights */}
      <Card sx={{ mb: 4, bgcolor: 'primary.lighter' }}>
        <CardHeader
          // avatar={
          //   <Avatar sx={{ bgcolor: 'primary.main', width: 56, height: 56 }}>
          //     <TrendingUp sx={{ fontSize: 32 }} />
          //   </Avatar>
          // }
          title={
            <Typography variant="h5" fontWeight={700}>
              Key Insights & Analysis
            </Typography>
          }
        />
        <CardContent>
          <Stack spacing={3}>
            {/* Insight 1 */}
            <Alert 
              severity="#ff1ff" 
              // icon={<CheckCircle />}
              sx={{ 
                borderRadius: 2,
                '& .MuiAlert-message': { width: '100%' }
              }}
            >
              <Typography variant="subtitle1" fontWeight={700} gutterBottom>
                1. Gradient Boosting performs the best overall
              </Typography>
              <Typography variant="body2">
                Achieves the highest validation accuracy (80.95%), test accuracy (75.21%), precision (81.24%), recall (80.95%), and F1-score (80.90%). 
                Captures complex, non-linear relationships better than the other models.
              </Typography>
            </Alert>

            {/* Insight 2 */}
            <Alert 
              severity="#fff" 
              // icon={<CheckCircle />}
              sx={{ 
                borderRadius: 2,
                '& .MuiAlert-message': { width: '100%' }
              }}
            >
              <Typography variant="subtitle1" fontWeight={700} gutterBottom>
                2. Random Forest is a close second
              </Typography>
              <Typography variant="body2">
                Slightly behind Gradient Boosting across all metrics (79.76% validation accuracy, 74.62% test accuracy). 
                Provides strong performance and good generalization.
              </Typography>
            </Alert>

            {/* Insight 3 */}
            <Alert 
              severity="#fff" 
              // icon={<CheckCircle />}
              sx={{ 
                borderRadius: 2,
                '& .MuiAlert-message': { width: '100%' }
              }}
            >
              <Typography variant="subtitle1" fontWeight={700} gutterBottom>
                3. Logistic Regression underperforms
              </Typography>
              <Typography variant="body2">
                Lower scores across all metrics compared to ensemble models (72.50% validation accuracy, 72.38% test accuracy). 
                Useful as a baseline or for interpretability but limited in predictive power for dropout prediction.
              </Typography>
            </Alert>

            {/* Insight 4 */}
            <Alert 
              severity="#fff" 
              // icon={<CheckCircle />}
              sx={{ 
                borderRadius: 2,
                '& .MuiAlert-message': { width: '100%' }
              }}
            >
              <Typography variant="subtitle1" fontWeight={700} gutterBottom>
                4. Precision & Recall tradeoff
              </Typography>
              <Typography variant="body2">
                Both Gradient Boosting and Random Forest achieve a strong balance, reflected in higher F1-scores (80.90% and 79.78% respectively). 
                This is important since both identifying potential dropouts (recall) and avoiding false alarms (precision) are critical.
              </Typography>
            </Alert>

            {/* Conclusion */}
            <Paper 
              elevation={0}
              sx={{ 
                p: 3, 
                bgcolor: 'success.lighter',
                border: '2px solid',
                borderColor: 'success.main',
                borderRadius: 2
              }}
            >
              <Typography variant="h6" fontWeight={700} color="success.dark" gutterBottom>
                <EmojiEvents sx={{ verticalAlign: 'middle', mr: 1 }} />
                Conclusion
              </Typography>
              <Divider sx={{ my: 2, borderColor: 'success.main' }} />
              <Stack spacing={1.5}>
                <Typography variant="body2">
                  • <strong>Gradient Boosting</strong> is the most suitable choice for deployment with best overall performance.
                </Typography>
                <Typography variant="body2">
                  • <strong>Random Forest</strong> is a strong alternative if interpretability or computational efficiency is prioritized.
                </Typography>
                <Typography variant="body2">
                  • <strong>Logistic Regression</strong> can be retained as a quick, interpretable baseline model.
                </Typography>
              </Stack>
            </Paper>
          </Stack>
        </CardContent>
      </Card>

     
    </Box>
  );
}
