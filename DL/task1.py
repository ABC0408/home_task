from xml.etree import ElementTree
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
def readxml(name):
    tree = ElementTree.parse(name)
    dict = {}
    for elem in tree.iter(tag='table'):
        for table in elem.iter(tag='column'):
            if table.attrib['name'] not in dict:
                dict[table.attrib['name']] = []
                dict[table.attrib['name']].append(table.text)
            else:
                dict[table.attrib['name']].append(table.text)
    df = pd.DataFrame(dict)
    df = df.iloc[:, :].replace({'NULL': 0})
    if 'bank' in name:
        f = lambda r: int(r.sberbank) + int(r.vtb) + int(r.gazprom) + int(r.alfabank) + int(r.bankmoskvy) + int(
            r.raiffeisen) + int(r.uralsib) + int(r.rshb)
    else:
        f = lambda r: int(r.beeline) + int(r.mts) + int(r.megafon) + int(r.tele2) + int(r.rostelecom) + int(
            r.komstar) + int(r.skylink)
    df['labels'] = df.iloc[:, :].apply(f, axis=1)
    return df

df = readxml('tkk_train_2016.xml')
df_2 = readxml('bank_train_2016.xml')
df_3 = readxml('banks_test_etalon.xml')
dft = readxml('tkk_test_2016.xml')
dftt = readxml('banks_test_2016.xml')
dftt_2 = readxml('tkk_test_etalon.xml')
col = ['text', 'labels']
df_train = pd.concat([df[col], df_2[col]], ignore_index=True)
df_test = pd.concat([dft[col], dftt[col]], ignore_index=True)
df_test_eta = pd.concat([df_3[col], dftt_2[col]], ignore_index=True)
df_test = pd.concat([df_test[col], df_test_eta[col]], ignore_index=True)
print(df_train.shape, df_test.shape)
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2')
train_x = tfidf.fit_transform(df_train.text).toarray()
test_x = tfidf.fit_transform(df_train.text).toarray()
train_y = df_train.labels
test_y = df_test.labels
print(train_x.shape)

clf = LogisticRegression(C=0.1).fit(X=train_x, y=train_y)
pred = clf.predict(test_x)
n = 0
for i, ii in zip(pred, test_y):
    if i == ii:
        n += 1
print(n/len(pred) * 100)