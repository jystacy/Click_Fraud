# data engineering
## generate x_1s, x_30s, x_60s

import pandas as pd
import datetime
import time
import os
import multiprocessing
from multiprocessing import Process
print ('number of cpu: ', multiprocessing.cpu_count())

def func(df, q, i):

    row = df.iloc[i, :]

    sing_key_60s = ['app_60s', 'ip_60s', 'channel_60s', 'device_60s', 'os_60s']
    sing_dic_60s = {k: 0 for k in sing_key_60s}

    sing_key_30s = ['app_30s', 'ip_30s', 'channel_30s', 'device_30s', 'os_30s']
    sing_dic_30s = {k: 0 for k in sing_key_30s}

    sing_key_1s = ['app_1s', 'ip_1s', 'channel_1s', 'device_1s', 'os_1s']
    sing_dic_1s = {k: 0 for k in sing_key_1s}

    col_list = ['app', 'os', 'ip', 'device', 'channel']
    df_sub_60s = pd.DataFrame()
    df_sub_30s = pd.DataFrame()
    df_sub_1s = pd.DataFrame()

    for l in range(60):
        delta_time = datetime.timedelta(0,l)
        time = row['click_time'] - delta_time
        time_sub = df[df['click_time'] == time]
        df_sub_60s = df_sub_60s.append(time_sub)
        if l <= 30:
            df_sub_30s = df_sub_30s.append(time_sub)
        if l <= 1:
            df_sub_1s = df_sub_1s.append(time_sub)

    df_sub_1s = df_sub_1s[df_sub_1s['record'] < row['record']]
    df_sub_30s = df_sub_30s[df_sub_30s['record'] < row['record']]
    df_sub_60s = df_sub_60s[df_sub_60s['record'] < row['record']]

    for m in col_list:
        id = row[m]
        df_ssub_60s = df_sub_60s[df_sub_60s[m] == id]
        sing_dic_60s[m + '_60s'] = len(df_ssub_60s)
        df_ssub_30s = df_sub_30s[df_sub_30s[m] == id]
        sing_dic_30s[m + '_30s'] = len(df_ssub_30s)
        df_ssub_1s = df_sub_1s[df_sub_1s[m] == id]
        sing_dic_1s[m + '_1s'] = len(df_ssub_1s)

    df1 = pd.DataFrame([sing_dic_1s])
    df2 = pd.DataFrame([sing_dic_30s])
    df3 = pd.DataFrame([sing_dic_60s])
    df4 = df1.join(df2)
    df5 = df4.join(df3)
    row = row.to_frame().transpose().reset_index()
    df6 = row.join(df5)
    q.put(df6)


def main():
    # read da ta
    df = pd.read_csv('clickfraud.csv')
    df['click_time'] = df['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

    print('Parent process %s.' % os.getpid())
    start_time = time.time()

    step = 100
    start = 57007
    End = len(df)
    print 'Total lines: ', End

    # use multiprocess to boost processing
    for j in range(int((End-57007)/step) + 1):
        remain = End - start
        checked_step = min(step, remain)
        end = start + checked_step + 1
        iterable = range(start, end)
        start = end

        print 'Start Iterable', iterable[0], 'to', iterable[-1]

        q = multiprocessing.Queue()
        procs = []
        for i in iterable:
            proc = Process(target=func, args=(df, q, i,))
            procs.append(proc)
            proc.start()

        all_result = pd.DataFrame()
        result = pd.DataFrame()
        for k in procs:
            res = q.get()
            # print type(res), res
            result = result.append(res, ignore_index=True)
        q.close()

        all_result = all_result.append(result, ignore_index=True)
        if not os.path.exists('result_sing.csv'):
            with open('result_sing.csv', 'a+') as f:
                all_result.to_csv(f, index=None, header=True)
        else:
            with open('result_sing.csv', 'a+') as f:
                all_result.to_csv(f, index=None, header=False)

    end = time.time() - start_time
    print('Total time: %s' % end)

if __name__ == '__main__':
    main()