
#ckages, modules, pragmas ===
## ==================================

## === 3rd party ===
import pandas as pd
import numpy as np

## === data setup ===
countries = ['Russian Fed.', 'Norway', 'Canada', 'United States',
                  'Netherlands', 'Germany', 'Switzerland', 'Belarus',
                  'Austria', 'France', 'Poland', 'China', 'Korea', 
                  'Sweden', 'Czech Republic', 'Slovenia', 'Japan',
                  'Finland', 'Great Britain', 'Ukraine', 'Slovakia',
                  'Italy', 'Latvia', 'Australia', 'Croatia', 'Kazakhstan']

gold = [13, 11, 10, 9, 8, 8, 6, 5, 4, 4, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
silver = [11, 5, 10, 7, 7, 6, 3, 0, 8, 4, 1, 4, 3, 7, 4, 2, 4, 3, 1, 0, 0, 2, 2, 2, 1, 0]
bronze = [9, 10, 5, 12, 9, 5, 2, 1, 5, 7, 1, 2, 2, 6, 2, 4, 3, 1, 2, 1, 0, 6, 2, 1, 0, 1]

olympic_medal_counts = {'country_name': pd.Series(countries),
                             'gold'   : pd.Series(gold),
                             'silver' : pd.Series(silver),
                             'bronze' : pd.Series(bronze)}
medal_df = pd.DataFrame(olympic_medal_counts, columns=['country_name','gold','silver','bronze'])



medal_score = [4,2,1]
medal_df['points'] = np.dot(medal_df[['gold','silver','bronze']],medal_score)
olympic_points_df = medal_df[['country_name','points']]
