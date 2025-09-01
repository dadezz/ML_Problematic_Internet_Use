# data-web-mining-project
Progetto per il corso di Data and Web Mining, in coppia con @gabibbo03. 
* Specifica del progetto: `./2024 DWM project.pdf`
* Relazione finale del progetto: `./Report Finale.pdf`
* codici python: `./src/`. Nello specifico, `data_cleaning.ipynb` è il primo notebook su cui abbiamo lavorato, per poi spostarci su `main.ipynb`. gli altri file `.py` sono i moduli con le funzioni usate nei due notebook. Non è pensato per essere scalabile/modulare o comunque "buon" codice pulito, è funzionale al progetto e basta, le funzioni sono solo il copincolla del ccodice precedentemente scritto tutto di fila sul notebook.
* miglior modello di rete neurale trovato: `/src/neural_network_tuned_model.h5`. In `./my_keras_tuner` si possono trovare tutti gli altri modelli esplorati durante l'hyperparameter tuning automatico

# dataset description (Kaggle)
The Healthy Brain Network (HBN) dataset is a clinical sample of about five-thousand 5-22 year-olds who have undergone both clinical and research screenings. The objective of the HBN study is to find biological markers that will improve the diagnosis and treatment of mental health and learning disorders from an objective biological perspective. Two elements of this study are being used for this competition: physical activity data (wrist-worn accelerometer data, fitness assessments and questionnaires) and internet usage behavior data. The goal of this competition is to predict from this data a participant's Severity Impairment Index (sii), a standard measure of problematic internet use.

Note that this is a Code Competition, in which the actual test set is hidden. In this public version, we give some sample data in the correct format to help you author your solutions. The full test set comprises about 3800 instances.

The competition data is compiled into two sources, parquet files containing the accelerometer (actigraphy) series and csv files containing the remaining tabular data. The majority of measures are missing for most participants. In particular, the target sii is missing for a portion of the participants in the training set. You may wish to apply non-supervised learning techniques to this data. The sii value is present for all instances in the test set.

# HBN Instruments
The tabular data in train.csv and test.csv comprises measurements from a variety of instruments. The fields within each instrument are described in data_dictionary.csv. These instruments are:

- Demographics - Information about age and sex of participants.
- Internet Use - Number of hours of using computer/internet per day.
- Children's Global Assessment Scale - Numeric scale used by mental health clinicians to rate the general functioning of youths under the age of 18.
- Physical Measures - Collection of blood pressure, heart rate, height, weight and waist, and hip measurements.
- FitnessGram Vitals and Treadmill - Measurements of cardiovascular fitness assessed using the NHANES treadmill protocol.
- FitnessGram Child - Health related physical fitness assessment measuring five different parameters including aerobic capacity, muscular strength, muscular endurance, flexibility, and body composition.
- Bio-electric Impedance Analysis - Measure of key body composition elements, including BMI, fat, muscle, and water content.
- Physical Activity Questionnaire - Information about children's participation in vigorous activities over the last 7 days.
- Sleep Disturbance Scale - Scale to categorize sleep disorders in children.
- Actigraphy - Objective measure of ecological physical activity through a research-grade biotracker.
- Parent-Child Internet Addiction Test - 20-item scale that measures characteristics and behaviors associated with compulsive use of the Internet including compulsivity, escapism, and dependency.

Note in particular the field *PCIAT-PCIAT_Total.* The target *sii* for this competition is derived from this field as described in the data dictionary: *0 for None, 1 for Mild, 2 for Moderate, and 3 for Severe.* Additionally, each participant has been assigned a unique identifier id.


