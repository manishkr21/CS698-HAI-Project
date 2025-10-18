import React from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Grid,
  Typography,
  Avatar,
  Alert,
  AlertTitle,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Box,
  Chip,
  Stack,
  Divider
} from '@mui/material';
import { 
  Groups, 
  BarChart as BarChartIcon, 
  CheckCircle, 
  Warning,
  Balance,
  Assessment
} from '@mui/icons-material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';

export default function FairnessTab({ metrics }) {
  // Fairness evaluation data for all models
  const fairnessEvaluation = {
    'Random Forest': [
      {
        attribute: 'Gender',
        spd: -0.4103,
        eod: -0.0811,
        di: 1.3916,
        comment: 'Strong bias - Model favors unprivileged gender group (DI > 1).',
        severity: 'error'
      },
      {
        attribute: 'International',
        spd: 0.0208,
        eod: -0.4853,
        di: 0.9842,
        comment: 'Almost fair (SPD ≈ 0, DI ≈ 1) but recall gap exists.',
        severity: 'warning'
      },
      {
        attribute: 'Scholarship',
        spd: 0.6734,
        eod: 0.5098,
        di: 0.6319,
        comment: 'Model strongly favors scholarship holders (SPD > 0, DI < 0.8).',
        severity: 'error'
      }
    ],
    'Gradient Boosting': [
      {
        attribute: 'Gender',
        spd: -0.4045,
        eod: -0.0670,
        di: 1.4007,
        comment: 'Gender bias: favors unprivileged group.',
        severity: 'error'
      },
      {
        attribute: 'International',
        spd: -0.0222,
        eod: -0.3705,
        di: 1.0175,
        comment: 'Nearly balanced predictions, but recall gap remains.',
        severity: 'warning'
      },
      {
        attribute: 'Scholarship',
        spd: 0.6580,
        eod: 0.3970,
        di: 0.6294,
        comment: 'Bias towards scholarship holders.',
        severity: 'error'
      }
    ],
    'Logistic Regression': [
      {
        attribute: 'Gender',
        spd: -0.4228,
        eod: -0.2595,
        di: 1.4228,
        comment: 'High gender disparity (large SPD and DI > 1).',
        severity: 'error'
      },
      {
        attribute: 'International',
        spd: 0.1086,
        eod: -0.0865,
        di: 0.9148,
        comment: 'Mild disparity against international students.',
        severity: 'warning'
      },
      {
        attribute: 'Scholarship',
        spd: 0.6487,
        eod: 0.5096,
        di: 0.6337,
        comment: 'Strong bias towards scholarship holders.',
        severity: 'error'
      }
    ]
  };

  // Prepare data for visualization
  const prepareVisualizationData = () => {
    const allData = [];
    Object.keys(fairnessEvaluation).forEach(model => {
      fairnessEvaluation[model].forEach(metric => {
        allData.push({
          model: model,
          attribute: metric.attribute,
          spd: Math.abs(metric.spd),
          eod: Math.abs(metric.eod),
          di: metric.di
        });
      });
    });
    return allData;
  };

  // Get color based on severity
  const getSeverityColor = (severity) => {
    switch(severity) {
      case 'error': return 'error';
      case 'warning': return 'warning';
      case 'success': return 'success';
      default: return 'info';
    }
  };

  // Format metric value with color coding
  const formatMetric = (value, type) => {
    let color = 'default';
    if (type === 'spd' || type === 'eod') {
      const absValue = Math.abs(value);
      if (absValue < 0.1) color = 'success';
      else if (absValue < 0.3) color = 'warning';
      else color = 'error';
    } else if (type === 'di') {
      if (value >= 0.8 && value <= 1.25) color = 'success';
      else if ((value >= 0.7 && value < 0.8) || (value > 1.25 && value <= 1.4)) color = 'warning';
      else color = 'error';
    }
    return { value: value.toFixed(4), color };
  };

  return (
    <Box>
      {/* Header Alert */}
      <Alert severity="info" icon={<Balance />} sx={{ mb: 4, borderRadius: 2 }}>
        <AlertTitle sx={{ fontWeight: 700, fontSize: '1.1rem' }}>
          Fairness Evaluation - Results
        </AlertTitle>
        <Typography variant="body2">
          Comprehensive fairness analysis across protected attributes (Gender, International, Scholarship) for all models.
          Metrics include Statistical Parity Difference (SPD), Equal Opportunity Difference (EOD), 
          and Disparate Impact (DI).
        </Typography>
      </Alert>

      {/* Model Fairness Tables */}
      {Object.keys(fairnessEvaluation).map((modelName, idx) => (
        <Card key={idx} sx={{ mb: 4 }}>
          <CardHeader
            // avatar={
            //   <Avatar sx={{ bgcolor: idx === 0 ? 'primary.main' : idx === 1 ? 'warning.main' : 'success.main', width: 56, height: 56 }}>
            //     <Assessment sx={{ fontSize: 32 }} />
            //   </Avatar>
            // }
            title={
              <Typography variant="h6" fontWeight={500} textAlign={'center'} mt={2}>
                {`Fairness metrics for ${modelName} model`}
              </Typography>
            }
        
          />
          <CardContent>
            <TableContainer component={Paper} variant="outlined" sx={{ borderRadius: 2 }}>
              <Table>
                <TableHead sx={{ bgcolor: 'grey.100' }}>
                  <TableRow>
                    <TableCell><strong>Protected Attribute</strong></TableCell>
                    <TableCell align="center"><strong>Statistical Parity Difference</strong></TableCell>
                    <TableCell align="center"><strong>Equal Opportunity Difference</strong></TableCell>
                    <TableCell align="center"><strong>Disparate Impact</strong></TableCell>
                    <TableCell><strong>Comment</strong></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {fairnessEvaluation[modelName].map((row, rowIdx) => {
                    const spdFormat = formatMetric(row.spd, 'spd');
                    const eodFormat = formatMetric(row.eod, 'eod');
                    const diFormat = formatMetric(row.di, 'di');
                    
                    return (
                      <TableRow key={rowIdx} hover>
                        <TableCell>
                          <Chip 
                            label={row.attribute} 
                            color="primary" 
                            variant="outlined"
                            sx={{ fontWeight: 600 }}
                          />
                        </TableCell>
                        <TableCell align="center">
                          <Chip 
                            label={spdFormat.value} 
                            // color={spdFormat.color}
                            size="small"
                          />
                        </TableCell>
                        <TableCell align="center">
                          <Chip 
                            label={eodFormat.value} 
                            // color={eodFormat.color}
                            size="small"
                          />
                        </TableCell>
                        <TableCell align="center">
                          <Chip 
                            label={diFormat.value} 
                            // color={diFormat.color}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>
                          <Alert 
                            severity={getSeverityColor(row.severity)} 
                            sx={{ py: 0, px: 1 }}
                            icon={false}
                          >
                            <Typography variant="body2">
                              {row.comment}
                            </Typography>
                          </Alert>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      ))}

      {/* Comparative Visualization */}
      <Card sx={{ mb: 4 }}>
        <CardHeader
          // avatar={
          //   <Avatar sx={{ bgcolor: 'secondary.main', width: 56, height: 56 }}>
          //     <BarChartIcon sx={{ fontSize: 32 }} />
          //   </Avatar>
          // }
          title={
            <Typography variant="h5" fontWeight={700}>
              Fairness Metrics Comparison Across Models
            </Typography>
          }
          subheader="Comparative analysis of Statistical Parity Difference by protected attribute"
        />
        <CardContent>
          <Grid container spacing={3}>
            {['Gender', 'International', 'Scholarship'].map((attr, idx) => {
              const attrData = Object.keys(fairnessEvaluation).map(model => {
                const metric = fairnessEvaluation[model].find(m => m.attribute === attr);
                return {
                  model: model.replace(' ', '\n'),
                  SPD: Math.abs(metric.spd),
                  EOD: Math.abs(metric.eod)
                };
              });

              return (
                <Grid item xs={12} md={4} key={idx} width="32%">
                  <Paper variant="outlined" sx={{ p: 2, borderRadius: 2 }} >
                    <Typography variant="h6" gutterBottom align="center" color="primary">
                      {attr}
                    </Typography>
                    <Divider sx={{ mb: 2 }} />
                    <ResponsiveContainer width="100%" height={250}>
                      <BarChart data={attrData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                        <XAxis 
                          dataKey="model" 
                          tick={{ fontSize: 11 }}
                          interval={0}
                        />
                        <YAxis 
                          tick={{ fontSize: 11 }}
                          domain={[0, 1]}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'rgba(255, 255, 255, 0.95)', 
                            border: '2px solid #1976d2', 
                            borderRadius: '8px'
                          }} 
                        />
                        <Legend wrapperStyle={{ fontSize: '12px' }} />
                        <Bar dataKey="SPD" fill="#1976d2" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="EOD" fill="#ff9800" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </Paper>
                </Grid>
              );
            })}
          </Grid>
        </CardContent>
      </Card>

      {/* Key Findings */}
      <Card sx={{ bgcolor: 'info.lighter' }}>
        <CardHeader
          // avatar={
          //   <Avatar sx={{ bgcolor: 'info.main', width: 56, height: 56 }}>
          //     <CheckCircle sx={{ fontSize: 32 }} />
          //   </Avatar>
          // }
          title={
            <Typography variant="h5" fontWeight={500}>
              Key Findings & Recommendations
            </Typography>
          }
        />
        <CardContent>
          <Stack spacing={2.5} >
            <Alert severity="info" icon={<CheckCircle />}>
              <Typography variant="subtitle2" fontWeight={700}>Gender Bias (All Models)</Typography>
              <Typography variant="body2">
                All models show significant gender bias with <strong>DI values above 1.39</strong>, 
                indicating systematic favoritism toward one gender group.
              </Typography>
            </Alert>

            <Alert severity="info" icon={<CheckCircle  />}>
              <Typography variant="subtitle2" fontWeight={700}>International Student Fairness</Typography>
              <Typography variant="body2">
                Models perform relatively fair on international status (<strong>SPD close to 0</strong>), 
                but <strong>Equal Opportunity gaps exist</strong>, particularly in Random Forest (EOD = -0.4853).
              </Typography>
            </Alert>

            <Alert severity="info" icon={<CheckCircle/>}>
              <Typography variant="subtitle2" fontWeight={700}>Scholarship Holder Bias (All Models)</Typography>
              <Typography variant="body2">
                Strongest bias detected: All models heavily favor scholarship holders with 
                <strong> SPD &gt; 0.64 and DI &lt; 0.64</strong>. This may reflect genuine correlation with academic success 
                but requires careful interpretation.
              </Typography>
            </Alert>

            <Alert severity="info" icon={<CheckCircle  />}>
              <Typography variant="subtitle2" fontWeight={700}>Recommendation</Typography>
              <Typography variant="body2">
                • Use bias mitigation techniques for Gender and Scholarship attributes<br/>
                • Monitor Equal Opportunity metrics for International students<br/>
                • Consider ensemble approaches with fairness constraints<br/>
                • Regular audits of model decisions across protected groups
              </Typography>
            </Alert>
          </Stack>
        </CardContent>
      </Card>
    </Box>
  );
}
