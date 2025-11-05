import plotly.graph_objects as go

biomedparse = {
    'Bladder tumor': 15.91, 
    'Breast tumor': 26.38, 
    'Cervix tumor': 31.10, 
    'Prostate tumor': 17.74,
    'Uterus tumor': 43.31, 
    'Colon tumor': 66.40, 
    'Kidney tumor': 34.17, 
    'Liver tumor': 75.12, 
    'Lung tumor': 26.12, 
    'Pancreas tumor': 70.51, 
    'Heart': 89.50
}

biomedparse_finetuned = {
    'Bladder tumor': 70.36, 
    'Breast tumor': 71.57, 
    'Cervix tumor': 65.23, 
    'Prostate tumor': 44.10,
    'Uterus tumor': 72.56, 
    'Colon tumor': 72.29, 
    'Kidney tumor': 79.86, 
    'Liver tumor': 75.28, 
    'Lung tumor': 59.00, 
    'Pancreas tumor': 69.31, 
    'Heart': 72.33
}

biomedparse_lora = {
    'Bladder tumor': 69.81, 
    'Breast tumor': 70.36, 
    'Cervix tumor': 60.74, 
    'Prostate tumor': 43.59,
    'Uterus tumor': 69.59, 
    'Colon tumor': 69.02, 
    'Kidney tumor': 80.34, 
    'Liver tumor': 75.82, 
    'Lung tumor': 53.92, 
    'Pancreas tumor': 71.63, 
    'Heart': 77.54
}

categories = list(biomedparse.keys())

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
      r=list(biomedparse.values()),
      theta=categories,
      fill='toself',
      name='Product A'
))
fig.add_trace(go.Scatterpolar(
      r=list(biomedparse_finetuned.values()),
      theta=categories,
      fill='toself',
      name='Product B'
))
fig.add_trace(go.Scatterpolar(
      r=list(biomedparse_lora.values()),
      theta=categories,
      fill='toself',
      name='Product B'
))

fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 5]
    )),
  showlegend=False
)

fig.show()