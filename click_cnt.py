import pandas as pd
import datetime
import time
import os
import copy
import argparse

import multiprocessing
from multiprocessing import Process
print ('number of cpu: ', multiprocessing.cpu_count())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t', type=str, default='3', help='time slot: 3, 10')
    return parser.parse_args()

def func(df, q, i, args):
    row = df.iloc[i, :]

    inter_key = ['app_os_', 'ip_os_', 'channel_os_', 'device_os_', \
                 'os_app_', 'ip_app_', 'channel_app_', 'device_app_', \
                 'ip_device_', 'channel_device_', 'app_device_', 'os_device_', \
                 'app_ip_', 'os_ip_', 'device_ip_', 'channel_ip_', \
                 'app_channel_', 'os_channel_', 'ip_channel_', 'device_channel_']
    for ik in inter_key:
        ik = ik + args.t + 's'
    inter_dic = {k: [] for k in inter_key}

    sing_key = ['app_', 'ip_', 'channel_', 'device_', 'os_']
    for sk in sing_key:
        sk = sk + args.t + 's'
    sing_dic = {k: 0 for k in sing_key}

    col_list = ['app', 'os', 'ip', 'device', 'channel']
    df_sub = pd.DataFrame()

    for l in range(int(args.t)):
        delta_time = datetime.timedelta(0, l)
        time = row['click_time'] - delta_time
        time_sub = df[df['click_time'] == time]
        df_sub = df_sub.append(time_sub)
    df_sub = df_sub[df_sub['record'] < row['record']]

    for m in col_list:
        id = row[m]
        df_ssub = df_sub[df_sub[m] == id]
        sing_dic[m + '_' + args.t + 's'] = len(df_ssub)
        col_list2 = copy.deepcopy(col_list)
        col_list2.remove(m)
        for n in col_list2:
            inter_dic[n + '_' + m + '_' + args.t + 's'] = len(set(df_ssub[n]))

    df1 = pd.DataFrame([sing_dic])
    df2 = pd.DataFrame([inter_dic])
    df3 = df1.join(df2)
    row = row.to_frame().transpose().reset_index()
    df4 = row.join(df3)
    q.put(df4)


def main():
    args = parse_args()
    df = pd.read_csv('clickfraud.csv')
    df['click_time'] = df['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

    print'Parent process %s.' % os.getpid()
    start_time = time.time()

    step = 100 # set step for batch processing
    start = 57007 # start from index 57007
    End = len(df)
    print'Total lines: ', End

    for j in range(int((End - 57007) / step) + 1):
        remain = End - start
        checked_step = min(step, remain)
        end = start + checked_step + 1
        iterable = range(start, end)
        start = end

        print'Start Iterable', iterable[0], 'to', iterable[-1]

        # use multiprocessing to boost computation time
        q = multiprocessing.Queue()
        procs = []
        for i in iterable:
            proc = Process(target=func, args=(df, q, i, args,))
            procs.append(proc)
            proc.start()

        all_result = pd.DataFrame()
        result = pd.DataFrame()
        for k in procs:
            res = q.get()
            result = result.append(res, ignore_index=True) # grep results
        q.close()

        # save results to csv
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