#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# In[28]:


df_od_matrix = pd.read_csv("C:\\Users\\류석현\\Desktop\\미래기술첼린지\\2023 미래기술챌린지 과제가이드 (과제3 실시간 주문 대응 Routing 최적화) v4\\과제3 실시간 주문 대응 Routing 최적화 (od_matrix) 수정완료.csv", encoding="cp949")
df_veh_table = pd.read_csv("C:\\Users\\류석현\\Desktop\\미래기술첼린지\\2023 미래기술챌린지 과제가이드 (과제3 실시간 주문 대응 Routing 최적화) v4\\veh_table.csv", encoding="cp949")
df_veh_table2 = pd.read_csv("C:\\Users\\류석현\\Desktop\\미래기술첼린지\\2023 미래기술챌린지 과제가이드 (과제3 실시간 주문 대응 Routing 최적화) v4\\veh_table.csv", encoding="cp949")
df_orders_table2 = pd.read_csv("C:\\Users\\류석현\\Desktop\\미래기술첼린지\\2023 미래기술챌린지 과제가이드 (과제3 실시간 주문 대응 Routing 최적화) v4\\바뀐주문.csv", encoding ="cp949")
df_Terminals = pd.read_csv("C:\\Users\\류석현\\Desktop\\미래기술첼린지\\2023 미래기술챌린지 과제가이드 (과제3 실시간 주문 대응 Routing 최적화) v4\\과제3 실시간 주문 대응 Routing 최적화 (Terminals).csv", encoding = "cp949")
df_d_to_o = pd.read_csv("C:\\Users\\류석현\\Desktop\\미래기술첼린지\\우리가 따로 작업\\d(터미널)을 가장 가까운 o(터미널)데이터.csv")

########df_orders_table 가공 > 각 일, 배치, o 마다 주문건수 10개이하로 맞춰서 새로 가공함 #############
######## 위에 csv 불러오는 코드에서 df_orders_table2로 불러오는 것만 수정하면 아래 다 돌아가 ####### 

day = ["2023.5.1", "2023.5.2", "2023.5.3", "2023.5.4", "2023.5.5", "2023.5.6"]

group1 = (0, 1, 2, 3)

# 칼럼 이름들과 빈 데이터 프레임 생성
columns = ["주문ID", "하차지_위도", "하차지_경도", "착지ID", "CBM","하차가능시간_시작", "하차가능시간_종료", 
           "하차작업시간(분)", "터미널ID", "date", "Group", "마감기한", "출발시간", "클러스터"]

df_orders_table = pd.DataFrame(columns=columns)


for days in day :
    for groups in group1 :
        for o_index, row in df_Terminals.iterrows():
            o = row['ID']

            COF = (df_orders_table2["date"] == days) & (df_orders_table2["Group"] == groups) & (df_orders_table2["터미널ID"] == o)
            order_curr = df_orders_table2[COF].sort_values(['착지ID', 'CBM', '마감기한'], ascending=[False, False, True])

            if (len(order_curr) >= 10) :
                order_curr = order_curr[0:10]
                df_orders_table = pd.concat([df_orders_table, order_curr], ignore_index=True)
            else :
                df_orders_table = pd.concat([df_orders_table, order_curr], ignore_index=True)

# columns_to_drop = ["하차가능시간_시작", "하차가능시간_종료"]
# df_orders_table = df_orders_table.drop(columns=columns_to_drop)
                
df_orders_table


# In[29]:


#[ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
 
    # 아래에서 update_result_routes 함수 사용하는 부분도 한 군데 수정됨
    
def update_result_routes(once_route, current_route, df_d_to_o, start_center1, max_capacity1):
    
    global df_orders_table

    # 마지막으로 처리하는 주문의 도착지를 확인한 후, 해당 도착지와 가장 가까운 베이스 캠프(nearest_origin) 도출
    last_orderID = current_route[-1]
    final_terminal = once_route[-1]

    filtered_row = df_d_to_o[df_d_to_o['dterminal'] == final_terminal]
    nearest_origin = filtered_row['o'].values[0]
    
    # grouped_original = df_veh_table2.groupby(['StartCenter', 'MaxCapaCBM']).size().reset_index(name='Count')
    # grouped_now = df_veh_table.groupby(['StartCenter', 'MaxCapaCBM']).size().reset_index(name='Count')

    
    # 베이스캠프와 출발지 터미널 중에서 어디로 복귀해야 할지를 결정 (기준은 추후 수정해보기 / 현재는 30% )
    # 현재 차량의 CBM = max_capacity
    # 현재 차량의 StartCenter = start_center1
    # 출발지 터미널의 해당 차량 CBM에 해당하는 차량 보유율이 30% 이하로 떨어지면 출발지로 복귀
    # CBM별 30%로 했더니 주문 한 건이 처리가 안 돼서 40%로 늘려봤습니다. 안되면 20%으로 줄이는 것도 생각해보자
    if ((len(df_veh_table[(df_veh_table['MaxCapaCBM'] == max_capacity1) & (df_veh_table['StartCenter'] == start_center1)])) <= (0.4*len(df_veh_table2[(df_veh_table2['MaxCapaCBM'] == max_capacity1) & (df_veh_table2['StartCenter'] == start_center1)]))) :
        return_loation = start_center1
    else :
        return_loation = nearest_origin
    
    # 처리된 주문을 기존 데이터프레임에서 삭제
    for i in range (len(current_route)) :
        df_orders_table = df_orders_table[df_orders_table['주문ID'] != current_route[i]]
    

    return return_loation

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ]#
 
 # # TimeWindow 확인
def is_in_time_range(time_to_check, start_time, end_time):
    if start_time <= end_time:
        return start_time <= time_to_check <= end_time
    else:
        return time_to_check >= start_time or time_to_check <= end_time

# '마감기한' 칼럼 추가
def calculate_deadline(row):
    import pandas as pd
    if row['Group'] == 0:
        return row['date'] + pd.Timedelta(hours=72 - 1)
    elif row['Group'] == 1:
        return row['date'] + pd.Timedelta(hours=78 - 1)
    elif row['Group'] == 2:
        return row['date'] + pd.Timedelta(hours=84 - 1)
    elif row['Group'] == 3:
        return row['date'] + pd.Timedelta(hours=90 - 1)
    else:
        return None

def calculate_startline(row):
    import pandas as pd
    if row['Group'] == 0:
        return row['date'] + pd.Timedelta(hours=0)
    elif row['Group'] == 1:
        return row['date'] + pd.Timedelta(hours=6)
    elif row['Group'] == 2:
        return row['date'] + pd.Timedelta(hours=12)
    elif row['Group'] == 3:
        return row['date'] + pd.Timedelta(hours=18)
    else:
        return df_orders_table['date'] + pd.Timedelta(hours=0)

# def ord_change(df_orders_table) :
#     col = ("잔류현황", "VehicleID", "Sequence", "SiteCode", "ArrivalTime", "WaitingTime", "ServiceTime",
#        "DepartureTime", "Delivered")
#     filtering = pd.DataFrame(columns=col)
#     ORD_RESULT = pd.concat([df_orders_table, filtering], ignore_index=True)
#     ORD_RESULT.rename(columns={'주문ID': 'ORD_NO'}, inplace=True)
        
#     return ORD_RESULT

# def veh_change(df_veh_table) :
#     col = ( "Count", "Volume", "TravelDistance", "WorkTime", "TravelTime", "ServiceTime", "WaitingTime",
#        "TotalCost", "FixedCost", "VariableCost")
#     filtering = pd.DataFrame(columns=col)
#     VEH_RESULT = pd.concat([df_veh_table, filtering], ignore_index=True)
#     VEH_RESULT.rename(columns={'VehNum': 'VehicleID'}, inplace=True)
        
#     return VEH_RESULT

def time_minute(origin, destination, df_odmatrix) :
    try:
        # Origin, Destination 에따른 해당 인덱스를 true, false로 나타내는데, 한인덱스만 나올거
        n = ((df_odmatrix["origin"] == origin) & (df_odmatrix["Destination"] == destination))
        # true, false만 나타나므로 그걸 이용해서 true인 값을 n1에 저장
        n1 = df_odmatrix[n]
        # dc = 거리
        dc = n1.iloc[0, 2]
        # tm = time_minute 걸리는 시간
        tm = n1.iloc[0, 3]
        return tm
    
    #값이 안나오고 오류가 나올시 아래구문 탐
    except KeyError:
        print("해당 터미널 간의 거리 정보를 찾을 수 없습니다.")
        return None

    #travel_time 계산해주는 함수
def travel_time(o, once_route, nearest_origin) :
    global df_od_matrix
    
    veh_time = 0
    keep = []
    keep.append(o)
    keep.extend(once_route)

    #배달을 끝난후 차량의 시각을 저장.
    for i in range (2,len(keep)+1) :
        tm = time_minute(keep[i-2], keep[i-1], df_od_matrix)
        veh_time += tm
        if (i == len(keep)) :
             veh_time = veh_time + time_minute(keep[i-1], nearest_origin, df_od_matrix)
        
    return veh_time

# result_routes의 마지막 주문의 착지 ID를 확인해서 해당 터미널의 베이스캠프를 뱉어내는 함수



# 함수 호출 시 df_orders_table도 업데이트된 결과로 반환해야 함.


# In[30]:


df_veh_table['Used'] = 0

# date 칼럼을 pandas datetime으로 변환
df_orders_table['date'] = pd.to_datetime(df_orders_table['date'])

df_veh_table['현재시각'] = '2023-05-01 00:00:00'
df_veh_table['현재시각'] = pd.to_datetime(df_veh_table['현재시각'])

df_orders_table['d_near_origin'] = None

for _, row in df_d_to_o.iterrows():
    var = (df_orders_table['착지ID'] == row['dterminal'])
    df_orders_table.loc[var, 'd_near_origin'] = row['o']


# In[31]:


def process_orders_with_deadline(orders_df, veh_df, time_10):
    # 주문을 하차 가능 시간에 따라 정렬 -> TW 삭제했으므로 없어도 됨
    # orders_df = orders_df.sort_values(by='하차가능시간_시작')

    # 대신 목적지 순으로 정렬 ~ 같은 목적지가 먼저 할당되게 하기 위해
    orders_df = orders_df.sort_values(by="착지ID")

    # 결과를 저장할 리스트
    routes = []
    total_cost = 0
    max_cap_veh = set()
    processed_orders = set()
    slack = 5  # 나중에 최적화

    # 각 차량에 대해 처리
    for veh_index, veh_row in veh_df.iterrows():
        veh_num = veh_row['VehNum']
        max_capacity = veh_row['MaxCapaCBM']
        start_center = veh_row['StartCenter']
        variable_cost = veh_row['VariableCost']
        fixed_cost = veh_row['FixedCost']


        # 이미 처리된 라우트 건너뛰기
        if veh_num in max_cap_veh:
            continue

        current_capacity = 0  # 현재 차량의 용량
        current_location = start_center  # 현재 차량의 위치
        current_route = []  # 현재 차량의 경로
        current_cost = 0
        current_time = pd.to_datetime(time_10)  # 현재 시간을 주문 중 가장 빠른 출발시간으로 초기화
        once_route = []
        veh_time = 0
        time_log = [current_time]
        order_travel_distance = 0

#[ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#        
    
        # 주문 처리
        for index, order in orders_df.iterrows():
            order_id = order['주문ID']
            order_cbm = order['CBM']
            order_destination = order['착지ID']
            order_cluster = order['클러스터']
            order_time_start = order['하차가능시간_시작']
            order_time_end = order['하차가능시간_종료']
            order_deadline = order['마감기한']
                
            # 라우트가 비어 있으면 목적지 초기화
            if not current_route:
                route_destination = order_destination
                route_cluster = order_cluster
                route_time_start = order_time_start
                route_time_end = order_time_end
                
            # 이미 처리된 주문은 건너뛰기
            if order_id in processed_orders:
                continue
                
            # 다른 목적지는 건너 뛰기
            # if route_destination != order_destination:
            #     continue

            # '클러스터', '하차가능시간_시작', '하차가능시간_종료' 가 모두 같아야만 라우트에 담기 (해당 조건 만족 안 하면 건너 뛰기)
            if (route_cluster != order_cluster):
                continue
                
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ]#
        
            filtering = (df_od_matrix["Destination"] == order_destination) & (
                        df_od_matrix["origin"] == current_location)
            order_travel_time = float(df_od_matrix[filtering]["Time_minute"].iloc[0])  # pd.to_timedelta 로 인해서 데이터프레임 형식으로 나감...
            order_travel_distance = float(df_od_matrix[filtering]["Distance_km"].iloc[0])

            # 주문이 처리 가능한지 확인(CBM 상한을 안 넘는지 & Time Window 안에 있는지)
            if current_capacity + order_cbm <= max_capacity:
                # if order_time_start <= '23:59:59' or order_time_end >= '00:00:00':
                # 주문 처리
                current_capacity += order_cbm
                current_route.append(order_id)
                current_time = pd.to_datetime(time_10) + pd.to_timedelta(order_travel_time, unit='m')
                once_route.append(order_destination)
                time_log.append(current_time)

                # 현재 차량의 위치를 도착지로 업데이트 (목적지가 이제 달라지므로 활성화 시킴)
                current_location = order_destination

#[ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#   

        # 현재 차량의 경로와 발생 비용을 결과 리스트에 추가, 적절한 배차 상태에 있는지 확인
        if current_route:
            # 차량이 max 상태인지 확인
            if current_capacity <  max_capacity - slack:
                fit_cbm = min([x for x in veh_cbm if x >= current_capacity])
                for veh_index, veh_row in veh_df[veh_df['MaxCapaCBM'] == fit_cbm].iterrows():
                    veh_num = veh_row['VehNum']
                    fixed_cost = veh_row['FixedCost']
                    variable_cost = veh_row['VariableCost']
                    # 이미 처리된 라우트 건너뛰기
                    if veh_num in max_cap_veh:
                        continue
                    else:
                        break

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ]#           
  
            ## 현재 차량의 경로와 발생 비용을 결과 리스트에 추가 
        if once_route:
            # current_route의 'D_'뒷 부분 숫자가 작은 순으로 정렬
            once_route.sort(key=lambda x: int(x.split('_')[1]))  
            #복귀지점 반환받고, 처리된 주문 삭제 함수
            nearest_origin = update_result_routes(once_route, current_route, df_d_to_o, start_center, max_capacity)
            #시작 - 하차 - 복귀 까지 travel_time 구하는 함수
            veh_time = travel_time(o, once_route, nearest_origin)
            
            # time-log에 복귀 추가
            time_log.append(pd.to_datetime(time_10) + pd.to_timedelta(len(once_route) * 60 + veh_time, unit='m'))
            
            #차량의 현재시각 update
            df_veh_table.loc[df_veh_table['VehNum'] == veh_num, '현재시각'] = pd.to_datetime(time_10) + pd.to_timedelta(
                len(once_route) * 60 + veh_time, unit='m')
            

            ### 2. nearest_origin으로 복귀한 차량의 travel distance
            fi = (df_od_matrix["origin"] == once_route[-1]) & (df_od_matrix["Destination"] == nearest_origin)
            order_travel_distance += df_od_matrix.loc[fi, "Distance_km"].iloc[0]
            
            # once_route에 nearest_origin 붙이기
            once_route.append(nearest_origin)

 #[ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
            # 차량의 (현재) 위치 업데이트
            df_veh_table.loc[df_veh_table['VehNum'] == veh_num, 'StartCenter'] = nearest_origin

            #차량의 사용 여부 update
            df_veh_table.loc[df_veh_table['VehNum'] == veh_num, 'Used'] = 1
            
 #ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ]#            
            current_cost = variable_cost * order_travel_distance
            total_cost += current_cost
            routes.append((veh_num, start_center, current_route, once_route, current_cost, current_capacity, current_time,
                           order_travel_distance, fixed_cost, variable_cost, veh_time, time_log))

            max_cap_veh.add(veh_num)
            
        for ord in current_route:
            processed_orders.add(ord)

#     for i in max_cap_veh:
#         print(i)

    return routes, total_cost


# In[32]:


def process_orders_with_deadline2(orders_df, veh_df, time_10):
    # 주문을 하차 가능 시간에 따라 정렬
    orders_df = orders_df.sort_values(by='하차가능시간_시작')

    # 결과를 저장할 리스트
    routes = []
    total_cost = 0
    max_cap_veh = set()
    processed_orders = set()
    slack = 5  # 나중에 최적화

    # 각 차량에 대해 처리
    for veh_index, veh_row in veh_df.iterrows():
        veh_num = veh_row['VehNum']
        max_capacity = veh_row['MaxCapaCBM']
        start_center = veh_row['StartCenter']
        variable_cost = veh_row['VariableCost']
        fixed_cost = veh_row['FixedCost']


        # 이미 처리된 라우트 건너뛰기
        if veh_num in max_cap_veh:
            continue

        current_capacity = 0  # 현재 차량의 용량
        current_location = start_center  # 현재 차량의 위치
        current_route = []  # 현재 차량의 경로
        current_cost = 0
        current_time = pd.to_datetime(time_10)  # 현재 시간을 주문 중 가장 빠른 출발시간으로 초기화
        once_route = []
        veh_time = 0
        time_log = [current_time]
        order_travel_distance = 0


#[ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#        
    
    
        # 주문 처리
        for index, order in orders_df.iterrows():
            order_id = order['주문ID']
            order_cbm = order['CBM']
            order_destination = order['착지ID']
            order_cluster = order['클러스터']
            order_time_start = order['하차가능시간_시작']
            order_time_end = order['하차가능시간_종료']
            order_deadline = order['마감기한']
                
            # 라우트가 비어 있으면 목적지 초기화
            if not current_route:
                #route_destination = order_destination
                route_cluster = order_cluster
                route_time_start = order_time_start
                route_time_end = order_time_end
                
            # 이미 처리된 주문은 건너뛰기
            if order_id in processed_orders:
                continue
                
#             # 다른 목적지는 건너 뛰기
#             if route_destination != order_destination:
#                 continue
                
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ]#
            # 도착지 터미널이 od_matrix_df에 있는지 확인
            if order_destination in df_od_matrix["Destination"].values and current_location in df_od_matrix[
                "origin"].values:
                filtering = (df_od_matrix["Destination"] == order_destination) & (
                            df_od_matrix["origin"] == current_location)
                order_travel_time = float(df_od_matrix[filtering]["Time_minute"].iloc[0])  # pd.to_timedelta 로 인해서 데이터프레임 형식으로 나감...
                order_travel_distance = float(df_od_matrix[filtering]["Distance_km"].iloc[0])

            # 주문이 처리 가능한지 확인(CBM 상한을 안 넘는지 & Time Window 안에 있는지)
            if current_capacity + order_cbm <= max_capacity:
                # if (current_capacity + order_cbm <= max_capacity) and (current_time <= pd.to_datetime(order_time_end)):
                if order_time_start <= '23:59:59' or order_time_end >= '00:00:00':
                    # 주문 처리
                    current_capacity += order_cbm
                    current_route.append(order_id)
                    current_time = pd.to_datetime(time_10) + pd.to_timedelta(order_travel_time, unit='m')
                    once_route.append(order_destination)
                    time_log.append(current_time)

#[ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#  
                        
                    # 현재 차량의 위치를 도착지로 업데이트 (목적지가 이제 달라지므로 활성화 시킴)
                    current_location = order_destination

                    # 처리된 주문 기록
                    # processed_orders.add(order_id)
                    

 
        # 현재 차량의 경로와 발생 비용을 결과 리스트에 추가, 적절한 배차 상태에 있는지 확인
        if current_route:
            # 차량이 max 상태인지 확인
            if current_capacity <  max_capacity - slack:
                fit_cbm = min([x for x in veh_cbm if x >= current_capacity])
                for veh_index, veh_row in veh_df[veh_df['MaxCapaCBM'] == fit_cbm].iterrows():
                    veh_num = veh_row['VehNum']
                    fixed_cost = veh_row['FixedCost']
                    variable_cost = veh_row['VariableCost']
                    # 이미 처리된 라우트 건너뛰기
                    if veh_num in max_cap_veh:
                        continue
                    else:
                        break
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ]#           
  
            ## 현재 차량의 경로와 발생 비용을 결과 리스트에 추가 
        if once_route:
            # current_route의 'D_'뒷 부분 숫자가 작은 순으로 정렬
            once_route.sort(key=lambda x: int(x.split('_')[1]))  
            #복귀지점 반환받고, 처리된 주문 삭제 함수
            nearest_origin = update_result_routes(once_route, current_route, df_d_to_o, start_center, max_capacity)
            #시작 - 하차 - 복귀 까지 travel_time 구하는 함수
            veh_time = travel_time(o, once_route, nearest_origin)
            
            # time-log에 복귀 추가
            time_log.append(pd.to_datetime(time_10) + pd.to_timedelta(len(once_route) * 60 + veh_time, unit='m'))
            
            #차량의 현재시각 update
            df_veh_table.loc[df_veh_table['VehNum'] == veh_num, '현재시각'] = pd.to_datetime(time_10) + pd.to_timedelta(
                len(once_route) * 60 + veh_time, unit='m')
            

            ### 2. nearest_origin으로 복귀한 차량의 travel distance
            fi = (df_od_matrix["origin"] == once_route[-1]) & (df_od_matrix["Destination"] == nearest_origin)
            order_travel_distance += df_od_matrix.loc[fi, "Distance_km"].iloc[0]
            
            

            # once_route에 nearest_origin 붙이기
            once_route.append(nearest_origin)

            

 #[ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
            # 차량의 (현재) 위치 업데이트
            df_veh_table.loc[df_veh_table['VehNum'] == veh_num, 'StartCenter'] = nearest_origin
        
            #차량의 사용 여부 update
            df_veh_table.loc[df_veh_table['VehNum'] == veh_num, 'Used'] = 1
            
 #ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ]#            
            current_cost = variable_cost * order_travel_distance
            total_cost += current_cost
            routes.append((veh_num, start_center, current_route, once_route, current_cost, current_capacity, current_time,
                           order_travel_distance, fixed_cost, variable_cost, veh_time, time_log))

            max_cap_veh.add(veh_num)
            
        for ord in current_route:
            processed_orders.add(ord)

#     for i in max_cap_veh:
#         print(i)

    return routes, total_cost


# In[33]:


from tqdm import tqdm
import time
from datetime import datetime, timedelta

# day = ["2023.5.1", "2023.5.2", "2023.5.3", "2023.5.4", "2023.5.5", "2023.5.6"]

group1 = (0, 1, 2, 3)

days = ["2023-05-01", "2023-05-02", "2023-05-03", "2023-05-04", "2023-05-05", "2023-05-06"]
# days = ["2023-05-01"]

veh_cbm = [27, 33, 42, 51, 55]

# 주문결과 테이블
col = ("VehicleID", "Sequence", "ORD_NO", "SiteCode", "ArrivalTime", "WaitingTime", "ServiceTime",
       "DepartureTime", "Delivered")
ORD_RESULT = pd.DataFrame(columns=col)

# 배차결과 테이블
col = ("VehicleID", "Count", "Volume", "TravelDistance", "WorkTime", "TravelTime", "ServiceTime", "WaitingTime",
       "TotalCost", "FixedCost", "VariableCost")
VEH_RESULT = pd.DataFrame(columns=col)

# df_orders_table 데이터프레임의 모든 칼럼 가져오기
col = df_orders_table.columns

# 새롭게 쓸df에 모든 칼럼 추가(df_orders_table의 칼럼만 담고 모든 데이터없음)
ord_curr2 = pd.DataFrame(columns=col)

pd.set_option('display.max_columns', None) ## 모든 열을 출력한다.

start_days = day[0]
start_time = datetime(year=2023, month=4, day=30, hour=18, minute=0)  # 시작 시간을 설정
check_point = -6  ### 비상 사태에 노출된 주문을 확인하기 위한 변수 (36의 배수가 되면 비상 사태에 놓인 주문이 있는지 check함)

### 1. 주문처리시간 72시간으로 다시 돌림

divider = 36

###

# time10 추가
time_10 = datetime(year=2023, month=4, day=30, hour=23, minute=50)

result_routes_list=[]
a = 0


# In[34]:


# ### 1일만 돌렸을 때는 업데이트가 되는 걸 확인함.

# for g in group1:
#     a += 1
#     start_time += timedelta(hours=6)  # 시간을 6시간씩 증가
#     time_10 = start_time - timedelta(minutes=10)
#     check_point += 6  ### 비상 사태 확인용 변수를 6씩 증가 시킴 (나중에 36의 배수가 되면 비상 check)

#     for i in range(36) :
#         time_10 += timedelta(minutes=10)
#         for o_index, row in df_Terminals.iterrows():
#             o = row['ID']
#             COF = (df_orders_table["date"] == "2023-05-01") & (df_orders_table["Group"] == g) &  (df_orders_table["터미널ID"] == o)
#             order_curr = df_orders_table[COF].sort_values(['착지ID', 'CBM', '마감기한'], ascending=[False, False, True])

#             CVF =  ((df_veh_table["현재시각"] <= pd.Timestamp(time_10)) & (df_veh_table["StartCenter"] == o))
#             veh_curr = df_veh_table[CVF].sort_values(by=['Used','MaxCapaCBM'], ascending=[False, False])

#             result_routes, result_total_cost= process_orders_with_deadline(order_curr, veh_curr, time_10)

#             result_routes_list.append(result_routes)

#             Count = 0
#             Volume = 0
#             TravelDistance = 0
#             WorkTime = 0
#             TravelTime = 0
#             ServiceTime = 0
#             WaitingTime = 0
#             TotalCost = 0
#             FixedCost = 0
#             VariableCost = 0


#             for veh_num, sc, route, loc, cost, capacity, time, distance, fixed_cost, variable_cost, veh_time, t_log in result_routes:
#                 Count = len(route)
#                 Volume = capacity
#                 TravelDistance = distance
#                 TravelTime = veh_time
#                 ServiceTime = Count * 60
#                 WaitingTime = 0  # 모름, 수정
#                 WorkTime = TravelTime + ServiceTime + WaitingTime
#                 TotalCost = cost
#                 FixedCost = fixed_cost
#                 VariableCost = variable_cost * distance

#                 if VEH_RESULT[VEH_RESULT["VehicleID"] == veh_num].empty:
#                     new_row = [veh_num, Count, Volume, TravelDistance, WorkTime, TravelTime, ServiceTime,
#                                 WaitingTime, TotalCost + FixedCost, FixedCost, VariableCost]
#                     VEH_RESULT.loc[len(VEH_RESULT)] = new_row
#                 else:
#                     filt = VEH_RESULT['VehicleID'] == veh_num
#                     VEH_RESULT.loc[filt, 'Count'] += Count
#                     VEH_RESULT.loc[filt, 'Volume'] += Volume
#                     VEH_RESULT.loc[filt, 'TravelDistance'] += TravelDistance
#                     VEH_RESULT.loc[filt, 'WorkTime'] += WorkTime
#                     VEH_RESULT.loc[filt, 'TravelTime'] += TravelTime
#                     VEH_RESULT.loc[filt, 'ServiceTime'] += ServiceTime
#                     VEH_RESULT.loc[filt, 'WaitingTime'] += WaitingTime
#                     VEH_RESULT.loc[filt, 'TotalCost'] += TotalCost
# #                     VEH_RESULT.loc[filt, 'FixedCost'] += FixedCost
#                     VEH_RESULT.loc[filt, 'FixedCost'] = FixedCost
#                     VEH_RESULT.loc[filt, 'VariableCost'] += VariableCost

# print("차량 결과")
# print(VEH_RESULT)
watch = []


# In[35]:


# 주문 처리
progress_bar = tqdm(days, total=6, desc="Processing Orders")

for day in progress_bar :
    # print(day)
    for g in group1 :
        a += 1
        start_time += timedelta(hours=6)  # 시간을 6시간씩 증가
        time_10 = start_time - timedelta(minutes=10)
        check_point += 6  ### 비상 사태 확인용 변수를 6씩 증가 시킴 (나중에 36의 배수가 되면 비상 check)

        for i in range(36) :
            time_10 += timedelta(minutes=10)
            for o_index, row in df_Terminals.iterrows():
                o = row['ID']

                COF = (df_orders_table["date"] == day) & (df_orders_table["Group"] == g) &  (df_orders_table["터미널ID"] == o)
                order_curr = df_orders_table[COF].sort_values(['클러스터', '착지ID', 'CBM', '마감기한'], ascending=[False, False, False, True])

                CVF =  ((df_veh_table["현재시각"] <= pd.Timestamp(time_10)) & (df_veh_table["StartCenter"] == o))
                veh_curr = df_veh_table[CVF].sort_values(by=['Used','MaxCapaCBM'], ascending=[False, False])  

                if ((check_point % divider == 0) & (check_point != 0)) : ### 비상 사태를 check해야 하는 타이밍이 온다면
                    for ve, row2 in order_curr.iterrows() : 
                        deadline = row2['마감기한']
                        if time_10 > pd.to_datetime(deadline) - timedelta(hours=24) :
                            ord_curr2.loc[len(ord_curr2)] = row2
                            if not ord_curr2.empty :
                                ord_curr2 = ord_curr2.sort_values(['클러스터', '착지ID', '하차가능시간_시작','마감기한'], ascending=[False, False, False, True])

                                result_routes, result_total_cost= process_orders_with_deadline2(ord_curr2, veh_curr, time_10)

                                result_routes_list.append(result_routes)

                                Volume = 0
                                TravelDistance = 0
                                WorkTime = 0
                                TravelTime = 0
                                ServiceTime = 0
                                WaitingTime = 0
                                TotalCost = 0
                                FixedCost = 0
                                VariableCost = 0


                                for veh_num, sc, route, loc, cost, capacity, time, distance, fixed_cost, variable_cost, veh_time, t_log in result_routes:
                                    Count = len(route)
                                    Volume = capacity
                                    TravelDistance = distance
                                    TravelTime = veh_time
                                    ServiceTime = Count * 60
                                    WaitingTime = 0  # 모름, 수정
                                    WorkTime = TravelTime + ServiceTime + WaitingTime
                                    TotalCost = cost
                                    FixedCost = fixed_cost
                                    VariableCost = variable_cost * distance

                                    if VEH_RESULT[VEH_RESULT["VehicleID"] == veh_num].empty:
                                        new_row = [veh_num, Count, Volume, TravelDistance, WorkTime, TravelTime, ServiceTime,
                                                   WaitingTime, TotalCost+FixedCost, FixedCost, VariableCost]
                                        VEH_RESULT.loc[len(VEH_RESULT)] = new_row
                                    else:
                                        filt = VEH_RESULT['VehicleID'] == veh_num
                                        VEH_RESULT.loc[filt, 'Count'] += Count
                                        VEH_RESULT.loc[filt, 'Volume'] += Volume
                                        VEH_RESULT.loc[filt, 'TravelDistance'] += TravelDistance
                                        VEH_RESULT.loc[filt, 'WorkTime'] += WorkTime
                                        VEH_RESULT.loc[filt, 'TravelTime'] += TravelTime
                                        VEH_RESULT.loc[filt, 'ServiceTime'] += ServiceTime
                                        VEH_RESULT.loc[filt, 'WaitingTime'] += WaitingTime
                                        VEH_RESULT.loc[filt, 'TotalCost'] += TotalCost
                    #                     VEH_RESULT.loc[filt, 'FixedCost'] += FixedCost
                                        VEH_RESULT.loc[filt, 'FixedCost'] = FixedCost
                                        VEH_RESULT.loc[filt, 'VariableCost'] += VariableCost
                                continue   



                    # 우선, 비상 사태에 놓인 주문들을 추린다.(if time_10 > order_deadline - pd.Timedelta(hours=24):)
                    # 그러고 나서 해당 주문들을 처리한다. 이 때, 평소보다 널널한 제약 조건으로 처리한다. (비용이 좀 더 들더라도 비상 사태에 놓인주문은 우선적으로 처리해야 하니깐!)


                # 주문 처리 실행
                # result_routes, result_total_cost = process_orders_with_deadline(df_orders_table, df_veh_table, df_od_matrix)
                result_routes, result_total_cost= process_orders_with_deadline(order_curr, veh_curr, time_10)

                result_routes_list.append(result_routes)

                Count = 0
                Volume = 0
                TravelDistance = 0
                WorkTime = 0
                TravelTime = 0
                ServiceTime = 0
                WaitingTime = 0
                TotalCost = 0
                FixedCost = 0
                VariableCost = 0


                for veh_num, sc, route, loc, cost, capacity, time, distance, fixed_cost, variable_cost, veh_time, t_log in result_routes:
                    Count = len(route)
                    Volume = capacity
                    TravelDistance = distance
                    TravelTime = veh_time
                    ServiceTime = Count * 60
                    WaitingTime = 0  # 모름, 수정
                    WorkTime = TravelTime + ServiceTime + WaitingTime
                    TotalCost = cost
                    FixedCost = fixed_cost
                    VariableCost = variable_cost * distance

                    if VEH_RESULT[VEH_RESULT["VehicleID"] == veh_num].empty:
                        new_row = [veh_num, Count, Volume, TravelDistance, WorkTime, TravelTime, ServiceTime,
                                   WaitingTime, TotalCost+FixedCost, FixedCost, VariableCost]
                        VEH_RESULT.loc[len(VEH_RESULT)] = new_row
                    else:
                        filt = VEH_RESULT['VehicleID'] == veh_num
                        VEH_RESULT.loc[filt, 'Count'] += Count
                        VEH_RESULT.loc[filt, 'Volume'] += Volume
                        VEH_RESULT.loc[filt, 'TravelDistance'] += TravelDistance
                        VEH_RESULT.loc[filt, 'WorkTime'] += WorkTime
                        VEH_RESULT.loc[filt, 'TravelTime'] += TravelTime
                        VEH_RESULT.loc[filt, 'ServiceTime'] += ServiceTime
                        VEH_RESULT.loc[filt, 'WaitingTime'] += WaitingTime
                        VEH_RESULT.loc[filt, 'TotalCost'] += TotalCost
    #                     VEH_RESULT.loc[filt, 'FixedCost'] += FixedCost
                        VEH_RESULT.loc[filt, 'FixedCost'] = FixedCost
                        VEH_RESULT.loc[filt, 'VariableCost'] += VariableCost

        progress_bar.set_postfix(Order=f"{a}/{24}")

print("차량 결과")
print(VEH_RESULT)


# In[36]:


df_orders_table


# In[37]:


sum(VEH_RESULT["Count"])


# In[38]:


VEH_RESULT


# In[39]:


# 주문결과 테이블
col = ("VehicleID", "Sequence", "ORD_NO", "SiteCode", "ArrivalTime", "WaitingTime", "ServiceTime",
       "DepartureTime", "Delivered")
ORD_RESULT = pd.DataFrame(columns=col)

col2 = ("VehicleID", "Sequence", "ORD_NO", "SiteCode", "ArrivalTime", "WaitingTime", "ServiceTime",
       "DepartureTime", "Delivered")
ORD_PRED = pd.DataFrame(columns = col2)
empty_ORD = ORD_RESULT


for result_routes in result_routes_list:
    for veh_num, sc, route, loc, cost, capacity, time, distance, fixed_cost, variable_cost, veh_time, t_log in result_routes:
        sequence = 0
        empty_ORD = pd.DataFrame(columns=col)
        if sequence == 0:
            sequence += 1
            new_row = {"VehicleID" : veh_num,
                    "Sequence" : sequence,
                    "ORD_NO" : 'Null',
                    "SiteCode" : sc, # 터미널
                    "ArrivalTime" : str(t_log[0]), 
                    "WaitingTime" : 0, # 나중에 변경
                    "ServiceTime" : 0, # 나중에 변경
                    "DepartureTime" : str(t_log[0]), 
                    "Delivered" : 'Null'}
            empty_ORD.loc[empty_ORD.shape[0]] = new_row

        for index in range(len(route)+1):
            sequence += 1
            if loc[index][0] != 'O':
                new_row = {"VehicleID" : veh_num,
                        "Sequence" : sequence,
                        "ORD_NO" : route[index],
                        "SiteCode" : loc[index], 
                        "ArrivalTime" : str(t_log[index+1]), 
                        "WaitingTime" : (index)*60,
                        "ServiceTime" : 60,
                        "DepartureTime" : str(t_log[index+1]+timedelta(minutes=(index+1) * 60)), # 나중에 변경
                        "Delivered" : 'Null'}
                empty_ORD.loc[empty_ORD.shape[0]] = new_row
            else:
                new_row = {"VehicleID" : veh_num,
                    "Sequence" : 1,
                    "ORD_NO" : 'Null',
                    "SiteCode" : loc[index], # 터미널
                    "ArrivalTime" : str(t_log[index+1]), # 나중에 변경
                    "WaitingTime" : 0, # 나중에 변경
                    "ServiceTime" : 0, # 나중에 변경
                    "DepartureTime" : None, # 나중에 변경
                    "Delivered" : 'Null'}
                empty_ORD.loc[empty_ORD.shape[0]] = new_row

        ORD_PRED = pd.concat([ORD_PRED, empty_ORD], ignore_index = True)


# In[40]:


# if str(time_10)[11:] in ('0:00', '06:00', '12:00', '18:00', '24:00'):
# 시간 표시
time_10 = datetime(2023, 5, 1, 2, 0)

# 배송상태 업데이트
for _, row in ORD_PRED.iterrows():
    if row['ORD_NO'] != 'Null':
        if row['DepartureTime'] <= str(time_10):
            row['Delivered'] = 'Yes'
        else:
            row['Delivered'] = 'No'
    else:
        row['Delivered'] = 'Null'

# ORD_PRED에서 차량별로 처리 -> for 문
for veh in ORD_PRED['VehicleID'].unique():
    ex = ORD_PRED[ORD_PRED['VehicleID'] == veh]

    # DepartureTime 수정
    ex['DepartureTime'] = ex['DepartureTime'].fillna(method='bfill')
    ex = ex.drop_duplicates(subset=['SiteCode', 'DepartureTime'])

    ORD_RESULT = pd.concat([ORD_RESULT, ex], ignore_index=True)

filt = ORD_RESULT['SiteCode'].str.contains('O', case=False)
ORD_RESULT.loc[filt, 'WaitingTime'] = pd.to_datetime(ORD_RESULT.loc[filt, 'DepartureTime']) - pd.to_datetime(ORD_RESULT.loc[filt, 'ArrivalTime'])
  


# In[41]:


VEH_RESULT.to_csv("VEH_RESULT.csv")
ORD_RESULT.to_csv("ORD_RESULT.csv")


# In[42]:


len(df_orders_table)


# In[43]:


df_veh_table2[df_veh_table2["StartCenter"] == "O_50"]


# In[44]:


df_veh_table


# In[45]:


sum(VEH_RESULT["Count"])


# In[46]:


sum(VEH_RESULT["TotalCost"])


# In[47]:


df_orders_table


# In[48]:


ORD_RESULT


# In[49]:


VEH_RESULT


# In[ ]:




