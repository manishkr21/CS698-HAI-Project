import React from 'react';
import {
  Box,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Card,
  CardContent,
  Chip,
  Stack,
  Alert,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider
} from '@mui/material';
import {
  ExpandMore,
  HelpOutline,
  School,
  TrendingUp,
  Security,
  Psychology,
  Timeline,
  CheckCircle
} from '@mui/icons-material';

export default function HelpTab() {
  const faqs = [
    {
      category: 'Getting Started',
      icon: <School />,
      color: 'primary',
      questions: [
        {
          question: 'What is this platform for?',
          answer: 'This Student Dropout Analysis Platform uses advanced machine learning models to predict student dropout risk and analyze fairness across different demographic groups. It helps institutions identify at-risk students early and ensure equitable treatment.'
        },
        {
          question: 'How do I make a prediction?',
          answer: 'Go to the "Predict" tab, fill in the student information form with relevant details (demographics, academic performance, etc.), select a bias mitigation strategy if needed, and click "Predict Student Outcome". The system will provide probability scores for Dropout, Enrolled, and Graduate outcomes.'
        },
        {
          question: 'What data do I need to upload for training?',
          answer: 'You need a CSV file with 36 features plus a Target column. The features include student demographics, academic records, enrollment details, and socioeconomic indicators. The Target should be categorical (Dropout/Enrolled/Graduate).'
        }
      ]
    },
    {
      category: 'Model Performance',
      icon: <TrendingUp />,
      color: 'success',
      questions: [
        {
          question: 'Which model should I use?',
          answer: 'The "Best Model" is automatically selected based on validation accuracy. Typically, Gradient Boosting performs best for this task (~80% accuracy). However, you can compare all models in the "Performance & Fairness" tab to see detailed metrics for each.'
        },
        {
          question: 'What do the performance metrics mean?',
          answer: 'Accuracy shows overall correctness. Precision indicates how many predicted positives were correct. Recall shows how many actual positives were found. F1-Score balances precision and recall. The confusion matrix shows the distribution of predictions vs actual outcomes.'
        },
        {
          question: 'Why are there multiple models?',
          answer: 'Different models have different strengths. Logistic Regression is interpretable, Random Forest handles non-linear patterns, and Gradient Boosting often achieves the best performance. Having multiple models allows you to choose based on your priorities (accuracy vs interpretability).'
        }
      ]
    },
    {
      category: 'Fairness & Bias',
      icon: <Security />,
      color: 'warning',
      questions: [
        {
          question: 'What is bias mitigation?',
          answer: 'Bias mitigation reduces unfair treatment of protected groups (based on gender, age, etc.). Our platform offers baseline (no mitigation) and mitigated models that use fairness constraints to ensure more equitable predictions across demographic groups.'
        },
        {
          question: 'What fairness metrics should I monitor?',
          answer: 'Demographic Parity measures if prediction rates are equal across groups. Equalized Odds ensures equal true/false positive rates. Equal Opportunity focuses on equal true positive rates. Lower disparities (closer to 0) indicate fairer models.'
        },
        {
          question: 'When should I use bias mitigation?',
          answer: 'Use bias mitigation when you notice significant fairness disparities in the baseline model, or when making high-stakes decisions that could impact protected groups differently. Review the fairness metrics in the "Performance & Fairness" tab to decide.'
        },
        {
          question: 'What is the tradeoff with bias mitigation?',
          answer: 'Bias mitigation may slightly reduce overall accuracy (typically 1-3%) but significantly improves fairness across protected attributes. The Compare Models feature shows this tradeoff clearly, helping you make informed decisions.'
        }
      ]
    },
    {
      category: 'Understanding Predictions',
      icon: <Psychology />,
      color: 'secondary',
      questions: [
        {
          question: 'How do I interpret prediction probabilities?',
          answer: 'Each prediction shows three probabilities (Dropout, Enrolled, Graduate) that sum to 100%. Higher probability indicates stronger confidence. For example, 70% Dropout means the model is fairly confident the student will drop out.'
        },
        {
          question: 'What factors most influence predictions?',
          answer: 'Academic performance (grades, approved units), enrollment patterns (attendance, debtor status), demographic factors (age, previous qualifications), and socioeconomic indicators (scholarship, parents\' education/occupation) all contribute. First semester performance is particularly predictive.'
        },
        {
          question: 'Can I trust the predictions?',
          answer: 'Predictions should be used as decision support, not definitive answers. Model accuracy is ~80%, meaning 1 in 5 predictions may be incorrect. Always combine predictions with human judgment and consider the individual student\'s context.'
        }
      ]
    },
    {
      category: 'Features & Usage',
      icon: <Timeline />,
      color: 'info',
      questions: [
        {
          question: 'What is the Compare Models feature?',
          answer: 'This feature allows you to see how different bias mitigation strategies affect a single student\'s prediction. It shows side-by-side predictions from baseline and mitigated models, highlighting any differences in outcomes.'
        },
        {
          question: 'How often should I retrain models?',
          answer: 'Retrain when you have significant new data (e.g., end of semester), when model performance degrades, or when institutional policies change. Generally, retraining annually or semi-annually with updated data maintains accuracy.'
        },
        {
          question: 'What are the economic indicators?',
          answer: 'GDP, Unemployment Rate, and Inflation Rate capture macroeconomic conditions that may affect student persistence. These are typically set at enrollment and reflect the economic environment influencing the student\'s decision to continue education.'
        }
      ]
    }
  ];

  return (
    <Box>
    
      {/* Quick Tips */}
      <Alert severity="info" icon={<CheckCircle />} sx={{ mb: 4, borderRadius: 2 }}>
        <Typography variant="subtitle1" fontWeight={600} gutterBottom>Quick Start Tips</Typography>
        <List dense>
          <ListItem sx={{ py: 0 }}>
            <ListItemIcon sx={{ minWidth: 32 }}>
              <CheckCircle fontSize="small" color="info" />
            </ListItemIcon>
            <ListItemText primary="Start with the Predict tab to make individual student predictions" />
          </ListItem>
          <ListItem sx={{ py: 0 }}>
            <ListItemIcon sx={{ minWidth: 32 }}>
              <CheckCircle fontSize="small" color="info" />
            </ListItemIcon>
            <ListItemText primary="Review Performance & Fairness metrics to understand model behavior" />
          </ListItem>
          <ListItem sx={{ py: 0 }}>
            <ListItemIcon sx={{ minWidth: 32 }}>
              <CheckCircle fontSize="small" color="info" />
            </ListItemIcon>
            <ListItemText primary="Use Compare Models to see how bias mitigation affects predictions" />
          </ListItem>
          <ListItem sx={{ py: 0 }}>
            <ListItemIcon sx={{ minWidth: 32 }}>
              <CheckCircle fontSize="small" color="info" />
            </ListItemIcon>
            <ListItemText primary="Upload new training data to improve model accuracy over time" />
          </ListItem>
        </List>
      </Alert>

      {/* FAQ Sections */}
      {faqs.map((category, idx) => (
        <Box key={idx} sx={{ mb: 3 }}>
          <Stack direction="row" spacing={1.5} alignItems="center" sx={{ mb: 2 }}>
            {React.cloneElement(category.icon, { color: category.color, sx: { fontSize: 28 } })}
            <Typography variant="h6" color={`${category.color}.main`}>
              {category.category}
            </Typography>
            <Chip label={`${category.questions.length} questions`} size="small" color={category.color} variant="outlined" />
          </Stack>

          {category.questions.map((faq, qIdx) => (
            <Accordion
              key={qIdx}
              sx={{
                mb: 1,
                '&:before': { display: 'none' },
                borderRadius: 2,
                overflow: 'hidden',
                border: '1px solid',
                borderColor: 'divider'
              }}
            >
              <AccordionSummary
                expandIcon={<ExpandMore />}
                sx={{
                  '&:hover': { bgcolor: 'action.hover' },
                  '& .MuiAccordionSummary-content': { my: 1.5 }
                }}
              >
                <Typography variant="subtitle1" fontWeight={600}>
                  {faq.question}
                </Typography>
              </AccordionSummary>
              <AccordionDetails sx={{ pt: 0, pb: 2, px: 2, bgcolor: 'grey.50' }}>
                <Divider sx={{ mb: 2 }} />
                <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                  {faq.answer}
                </Typography>
              </AccordionDetails>
            </Accordion>
          ))}
        </Box>
      ))}

      {/* Contact Info */}
      <Card sx={{ mt: 4}}>
        <CardContent>
          <Typography variant="h6" gutterBottom>Need More Help?</Typography>
          <Typography variant="body2" sx={{ opacity: 0.95 }}>
            If you can't find the answer you're looking for, please contact your system administrator or 
            refer to the technical documentation for advanced configuration and troubleshooting.
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
}
