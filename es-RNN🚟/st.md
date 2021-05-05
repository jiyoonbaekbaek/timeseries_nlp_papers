```python
config = get_config('Daily') #25
```

```python
elif interval == 'Daily':
        config.update({
            #     RUNTIME PARAMETERS
            #'chop_val': 200,
            'chop_val': 0,
            'variable': "Daily",
            'dilations': ((1, 3), (7, 9)), ####needs change 
            'state_hsize': 50,
            'seasonality': 7,
            'input_size': 7,
            'output_size': 14,
            'level_variability_penalty': 50
        })
```

딕셔너리 update 메소드

```python
persons = [('김기수', 30), ('홍대길', 35), ('강찬수', 25)]
mydict = dict(persons)
 
mydict.update({'홍대길':33,'강찬수':26})
```

코드 내에서 사용한 데이터 

![캡처](https://user-images.githubusercontent.com/67775336/117126810-42c8e200-add6-11eb-9057-7f39c5097fd1.PNG)

```python
df = pd.read_csv('data/energy_daily.csv')
df = df['W']
df.max()
df.min()
# +
#df = (df - df.min())/(df.max() - df.min()) #model normalizes
# -
values = df.values   #[113884. ,..., ]
# data = []
# seq_len = 30
# for i in range(len(values)-seq_len+1):
#     #print(list(values[i:i+seq_len]))
#     data.append(list(values[i:i+seq_len].reshape(-1)))

# df = pd.DataFrame(data)

# df.head()
```

```python
x.reshape(-1)
>> array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
```

-1만 들어가면 1차원 배열을 반환한다.

