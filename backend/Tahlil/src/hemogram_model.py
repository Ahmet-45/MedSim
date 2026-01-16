import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

filepath = r"C:\Users\vıctus\Desktop\MedSim\Tahlil\data\IDA_dataset2.csv"
data = pd.read_csv(filepath)
data = data.loc[:, ["Hemoglobin", "RDW", "MCV", "Age", "Gender", "Anemia_Type"]]

print(data.head(), data.info())


metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)

synthesizer_2000 = CTGANSynthesizer(
    metadata, 
    enforce_rounding=False,
    epochs=4000,
    verbose=True
)
synthesizer_2000.fit(data)
synthetic_data = synthesizer_2000.sample(num_rows=len(data))
synthetic_data.head(4)
synthesizer_2000.save(r"C:\Users\vıctus\Desktop\MedSim\Tahlil\hemogram_model_IDA.pkl")
synthetic_data.to_csv(r"C:\Users\vıctus\Desktop\MedSim\Tahlil\cikti\cikti_hemogram_model.csv", index=False)

losses = synthesizer_2000.get_loss_values()

def _to_scalar(x):
    try:
        return x.item()
    except Exception:
        try:
            return float(x)
        except Exception:
            return x

losses = synthesizer_2000.get_loss_values()

# if column names differ, print(losses.columns) to inspect
for col in ("Generator Loss", "Discriminator Loss"):
    if col in losses:
        losses[col] = losses[col].apply(_to_scalar)

losses['Epoch'] = pd.to_numeric(losses.get('Epoch', losses.index), errors='coerce')

plt.figure(figsize=(30, 8))
plt.plot(losses['Epoch'], losses['Generator Loss'], label='Generator Loss')
plt.plot(losses['Epoch'], losses['Discriminator Loss'], label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Generator and Discriminator Loss Over Epochs')
plt.legend()
plt.show()


metadata_dict = metadata.to_dict()

from sdmetrics.reports.single_table import DiagnosticReport

diagnosticreport = DiagnosticReport()
diagnosticreport.generate(data, synthetic_data, metadata_dict)


from sdmetrics.reports.single_table import QualityReport

qualityreport = QualityReport()

qualityreport.generate(data, synthetic_data, metadata_dict, verbose = False)

print(qualityreport.get_score()) # Out: 0.8906988469261656
print(qualityreport.get_properties())
print(qualityreport.get_details("Column Pair Trends"))


data_corr = data.loc[:, ["Hemoglobin", "RDW", "MCV", "Age", "Gender", "Anemia_Type"]]
syn_corr = synthetic_data.loc[:, ["Hemoglobin", "RDW", "MCV", "Age", "Gender", "Anemia_Type"]]

corr_matrix_df1 = data_corr.corr()
corr_matrix_df2 = syn_corr.corr()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

sns.heatmap(corr_matrix_df1, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'label': 'Correlation'}, ax=axes[0])
axes[0].set_title('Correlation Matrix - Real Data')

sns.heatmap(corr_matrix_df2, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'label': 'Correlation'}, ax=axes[1])
axes[1].set_title('Correlation Matrix - Synthetic Data')

plt.tight_layout()
plt.show() # Figure 2


variables_to_compare = ["Hemoglobin", "RDW", "MCV", "Age", "Gender", "Anemia_Type"]
patterns = ['-', '--', ':', '-.']  # Add another pattern

warnings.filterwarnings("ignore")
plt.figure(figsize=(15, 8))

for i, variable in enumerate(variables_to_compare):
    sns.kdeplot(data[variable], label=f'{variable} - Real Data', shade=True, linestyle=patterns[i])
    sns.kdeplot(synthetic_data[variable], label=f'{variable} - Synthetic Data', shade=True, linestyle=patterns[i])

plt.xlabel('Variable Values')
plt.ylabel('Density')
plt.title('Distribution of Variables in Real and Synthetic Data')
plt.legend()
plt.show() # Figure 3


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
synthetic_data['SEX'].value_counts().plot(kind='bar', title='Synthetic Data - SEX')

plt.subplot(1, 2, 2)
data['SEX'].value_counts().plot(kind='bar', title='Real Data - SEX')

plt.show() # Figure 4


