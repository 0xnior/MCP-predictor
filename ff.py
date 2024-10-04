import pickle
import pandas as pd

pipe = pickle.load(open("IEX_model.pickle", 'rb'))
scaler = pickle.load(open("scaler.pickle", 'rb'))


d1 = pd.DataFrame([[2162.05,589.38,0.00,589.38,0.0,460.30,460.30,0.00,510.00]], columns=['Purchase', 'sbTotal', 'sbSolar', 'sbNonSolar', 'sbHydro',
                                'mcvTotal', 'mcvNonsolar', 'mcvHydro', 'fsvTotal'])


pipe.predict(scaler.transform(d1))