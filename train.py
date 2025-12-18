import pandas as pd
import jieba
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

# 1. Daten laden
# entweder utf-8 oder gb2312
try:
    df = pd.read_csv('data/waimai_10k.csv', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('data/waimai_10k.csv', encoding='gb2312')

print(f'Daten geladen: {len(df)} Zeilen')
print(df.head())

# Prüfen, wie die Spalten heißen (oft 'review' und 'label')

X_raw = df['review']
y = df['label'] # 0/1 1 für positiv

# 2. Daten mit Jieba vorbereiten
def prepare_text_for_sklearn(text):
    # jieba.lcut zerlegt die Sätze in Listen von jeweils einem Wort
    tokens = jieba.lcut(str(text))
    # ' '.join fügt die Wörter mit Leerzeichen zusammen
    return ' '.join(tokens)

# 3. Funktion auf Daten anwenden
X_prepared = X_raw.apply(prepare_text_for_sklearn) # apply ist wichtig, weil X_raw eine Pandas Series ist; apply() ist die Standardmethode in Pandas, um eine Funktion elementweise auf eine Series oder DataFrame-Spalte anzuwenden.

print('\nBeispiel Vorher: ', X_raw.iloc[0])
print('Beispiel Nachher: ', X_prepared.iloc[0])
print('-'*30)

####### Training und Pipeline #######
# CountVectorizer (zählt Wörter) und MultinomialNB (Naive Bayes - Standard für einfache Textklassifikation)
# Split in Test und Traingingsdaten
X_train, X_test, y_train, y_test = train_test_split(X_prepared, y, test_size=0.2, random_state=1)
# Pipeline bauen
# CountVectorizer nimmt die Strings mit Leerzeichen und wandelt sie in eine Matrix von Wortzählungen um
# MultinomialNB lernt daraus
clf = make_pipeline(CountVectorizer(), MultinomialNB())
clf.fit(X_train, y_train)

# 4. Modell evaluieren
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'\nModellgenauigkeit: {accuracy*100:.2f}%')

# 5. Modell speichern
joblib.dump(clf, 'model/waimai_model.pkl')
print("Modell gespeichert als 'waimai_model.pkl'")