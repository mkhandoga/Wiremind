import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

def r_square_loss(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return 1 - ( 1 - SS_res/(SS_tot + K.epsilon()))


poisson_loss = tf.keras.losses.Poisson()

from datetime import datetime
from dateutil import parser

def date_to_day_number(date):
    date_number = parser.parse(date)
    return date_number.timetuple().tm_yday

poisson_loss = tf.keras.losses.Poisson()

all_columns = ['current_price', 'days_before_departure', 'departure_date', 'direction',
       'train_number', 'demand', 'od_destination_time_year',
       'od_destination_time_month', 'od_destination_time_week',
       'od_destination_time_day', 'od_destination_time_weekday',
       'od_destination_time_hour', 'od_destination_time_minute',
       'od_origin_time_hour', 'od_origin_time_minute', 'od_time_travel']

columns_ready = ['current_price','days_before_departure', 'od_destination_time_month', 
                 'od_destination_time_week', 'od_destination_time_day', 'od_destination_time_weekday',
                'od_time_travel']

scalable_columns = columns_ready + ["od_destination_time_hourmin","od_origin_time_hourmin"]
nonscalable_columns = ["direction_bool" , "demand", "of_holiday", "unof_holiday","date_numerical"]

input_features = scalable_columns + ["direction_bool", "of_holiday", "unof_holiday"]

official_holidays_2018 = ["2018-04-02", "2018-05-01", "2018-05-08", "2018-05-10",
                         "2018-05-21", "2018-07-14", "2018-08-15", "2018-11-01",
                         "2018-11-11", "2018-12-25"]
unofficial_holidays_2018 = ["2018-01-06", "2018-02-02", "2018-02-13","2018-02-14" ,"2018-03-04",
                         "2018-03-17", "2018-05-16", "2018-05-27", "2018-06-17",
                         "2018-06-21", "2018-06-24","2018-09-19","2018-10-07", "2018-12-02"]
official_holidays_2019 = ["2019-04-22", "2019-05-01", "2019-05-08", "2019-05-30",
                         "2019-06-10", "2019-07-14", "2019-08-15", "2019-11-01",
                         "2019-11-11", "2019-12-25"]
unofficial_holidays_2019 = ["2019-01-06", "2019-02-02", "2019-02-14", "2019-03-03",
                         "2019-03-05", "2019-03-17", "2019-05-06", "2019-05-26",
                         "2019-06-16", "2019-06-21","2019-06-24","2019-10-06", "2019-10-09","2019-12-22"]

def features_preparation(all_features):
	if (all_features.od_destination_time_year.unique()[0] == "2018"):
		print ("It's for 2018")
		all_features["of_holiday"] =  all_features.departure_date.isin(official_holidays_2018)
		all_features["unof_holiday"] =  all_features.departure_date.isin(official_holidays_2018)
	else: 
		if (all_features.od_destination_time_year.unique()[0] == "2019"):
			print ("It's for 2019")
			all_features["of_holiday"] =  all_features.departure_date.isin(official_holidays_2019)
			all_features["unof_holiday"] =  all_features.departure_date.isin(official_holidays_2019)
		else:
			all_features["of_holiday"] =  0
			all_features["unof_holiday"] =  0

	all_features["od_destination_time_hourmin"] = 60*all_features["od_destination_time_hour"] + all_features["od_destination_time_minute"]
	all_features["od_origin_time_hourmin"] = 60*all_features["od_origin_time_hour"] + all_features["od_origin_time_minute"]
	all_features["direction_bool"] = (all_features['direction'] == "outbound").astype(int)
	all_features['date_numerical'] = all_features.departure_date.apply(date_to_day_number)
	return all_features

def features_scale(all_features):
	sc = MinMaxScaler()
	features_scaled = all_features [scalable_columns] 
	features_scaled.loc[:, scalable_columns] = sc.fit_transform(features_scaled.loc[:, scalable_columns])
	features_scaled[nonscalable_columns] = all_features[nonscalable_columns]
	return features_scaled
