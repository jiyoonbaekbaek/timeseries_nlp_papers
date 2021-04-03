## 추상 메소드가 구현된 GenericDataFormatter 을 상속받은 클래스들 

### 1) electricity.py 

**필요한 모듈 가져오기** 

```python
import data_formatters.base
import libs.utils as utils
import pandas as pd
import sklearn.preprocessing
```

**GenericDataFormatter 에서 정의해준 것 변수에 넣어주기** 

```python
GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes
```

Remind) 

TFT 에서 시계열 데이터 타입

- 실수형
- 카테고리형
- 날짜형

TFT 에서 인풋 타입

- y 값 (target)
- 측정 값
- 측정 X, 기본 값
- 메타 데이터
- ID
- 시간 (t)

**본격 GenericDataFormatter 을 물려받은 ElectricityFormatter 정의해주기** 

```python
class ElectricityFormatter(GenericDataFormatter):
```

```
  _column_definition = [
      ('id', DataTypes.REAL_VALUED, InputTypes.ID),
      ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.TIME),
      ('power_usage', DataTypes.REAL_VALUED, InputTypes.TARGET),
      ('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('categorical_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
  ]
```

정리해주면 electiricy data 는 

| id     | hours_from_start | power_usage  | hour    | day_of_week | hours_from_start | categorical_id |
| ------ | ---------------- | ------------ | ------- | ----------- | ---------------- | -------------- |
| 실수형 | 실수형           | 실수형       | 실수형  | 실수형      | 실수형           | 카테고리형     |
| ID     | 시간(t)          | y값 (target) | 기본 값 | 기본 값     | 기본 값          | 메타데이터     |

이렇게 구성되어있습니다. 

**ElectricityFormatter 에서 사용할 속성들**

```python
def __init__(self):
    """Initialises formatter."""

    self.identifiers = None
    self._real_scalers = None
    self._cat_scalers = None
    self._target_scaler = None
    self._num_classes_per_cat_input = None
    self._time_steps = self.get_fixed_params()['total_time_steps']
```

- identifiers
- _real_scalers
- _cat_scalers
- _target_scaler
- _num_classes_per_cat_input
- _time_steps 

**GenericDataFormatter 에 정의되었던 추상 메소드 **

remind ) 

- [x] set_scalers : calibrates scalers using the data supplied
- [x] transform_inputs : perform feature transformation
- [x] format_predictions : reverts any normalization to give predictions in original scale 
- [x] split_data : performs the default train, validation and test splits
- [x] get_fixed_params : defines the fixed parameters used by model for training 

**1 ) split_data** 

: splits data frame into training-validation-test data frames 

```python
def split_data(self, df, valid_boundary=1315, test_boundary=1339):
    """Splits data frame into training-validation-test data frames.
    This also calibrates scaling object, and transforms data for each split.
    Args:
      df: Source data frame to split.
      valid_boundary: Starting year for validation data
      test_boundary: Starting year for test data
    Returns:
      Tuple of transformed (train, valid, test) data.
    """

    print('Formatting train-valid-test splits.')

    index = df['days_from_start']
    train = df.loc[index < valid_boundary]
    valid = df.loc[(index >= valid_boundary - 7) & (index < test_boundary)]
    test = df.loc[index >= test_boundary - 7]

    self.set_scalers(train)

    return (self.transform_inputs(data) for data in [train, valid, test])
```

days_from_start 이다보니까 깔끔하기 일주일 간격으로 끊기 위해서 boundary - 7 을 해준 것 같습니다. 

**2 ) set_scalers** 

: calibrate scalers using the data supplied

input type 이 id , target 인 열, data type 이 실수형인 열 쫙 다 뽑아오기

```python
def set_scalers(self, df):
    """Calibrates scalers using the data supplied.
    Args:
      df: Data to use to calibrate scalers.
    """
    print('Setting scalers with training data...')

    column_definitions = self.get_column_definition()
    id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                   column_definitions)
    target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                       column_definitions)

    # Format real scalers
    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME})
```

``` python
# Initialise scaler caches
    self._real_scalers = {}
    self._target_scaler = {}
    identifiers = []
    for identifier, sliced in df.groupby(id_column):

      if len(sliced) >= self._time_steps:

        data = sliced[real_inputs].values
        targets = sliced[[target_column]].values
        self._real_scalers[identifier] \
      = sklearn.preprocessing.StandardScaler().fit(data)

        self._target_scaler[identifier] \
      = sklearn.preprocessing.StandardScaler().fit(targets)
      identifiers.append(identifier)
```

- df.groupby 

인자로 넣어준 칼럼을 기준으로 칼럼 값이 같은 것들끼리 그룹을 만들어 줌.

identifier : 각 그룹의 key (즉 id)

sliced : id 가 같은 그룹 내 아이템들을 담고 있는 실질적 group 

- sklearn.preprocessing

평균 0, 분산 1로 조정 

fit(data) : rescaling 해주기 위해서 데이터를 넣어주는 과정 (fit -> transform 의 순서)

- get_column_definition 리마인드

  ```python
  def get_column_definition(self):
      """"Returns formatted column definition in order expected by the TFT."""
  
      column_definition = self._column_definition
  
      # Sanity checks first.
      # Ensure only one ID and time column exist
      def _check_single_column(input_type):
  
        length = len([tup for tup in column_definition if tup[2] == input_type])
  
        if length != 1:
          raise ValueError('Illegal number of inputs ({}) of type {}'.format(
              length, input_type))
  
      _check_single_column(InputTypes.ID)
      _check_single_column(InputTypes.TIME)
  ```

```python
# Format categorical scalers
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    categorical_scalers = {}
    num_classes = []
    for col in categorical_inputs:
      # Set all to str so that we don't have mixed integer/string columns
      srs = df[col].apply(str)
      categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
          srs.values)
      num_classes.append(srs.nunique())

    # Set categorical scaler outputs
    self._cat_scalers = categorical_scalers
    self._num_classes_per_cat_input = num_classes

    # Extract identifiers in case required
    self.identifiers = identifiers
```

- LabelEncoder (범주형 -> 수치형)

[범주형 데이터 변환](https://mizykk.tistory.com/10)

- nunique( ) : 유니크한 value 개수 



**3) transform_inputs**

: performs feature transformations 

sklearn 의 transform 메소드를 통해서 본격 값들을 scaling 해주는 메소드

```python
def transform_inputs(self, df):
    """Performs feature transformations.
    This includes both feature engineering, preprocessing and normalisation.
    Args:
      df: Data frame to transform.
    Returns:
      Transformed data frame.
    """

    if self._real_scalers is None and self._cat_scalers is None:
      raise ValueError('Scalers have not been set!')

    # Extract relevant columns
    column_definitions = self.get_column_definition()
    id_col = utils.get_single_col_by_input_type(InputTypes.ID,
                                                column_definitions)
    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME})
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    # Transform real inputs per entity
    df_list = []
    for identifier, sliced in df.groupby(id_col):

      # Filter out any trajectories that are too short
      if len(sliced) >= self._time_steps:
        sliced_copy = sliced.copy()
        sliced_copy[real_inputs] = self._real_scalers[identifier].transform(
            sliced_copy[real_inputs].values)
        df_list.append(sliced_copy)

    output = pd.concat(df_list, axis=0)

    # Format categorical inputs
    for col in categorical_inputs:
      string_df = df[col].apply(str)
      output[col] = self._cat_scalers[col].transform(string_df)

    return output
```

- pd.concat 

  axis=0 일 경우 ⬇️ 행 방향으로 합침 !



**4) format_predictions** 

: reverts any normalisation to give predictions in original scale

```python
def format_predictions(self, predictions):
    """Reverts any normalisation to give predictions in original scale.
    Args:
      predictions: Dataframe of model predictions.
    Returns:
      Data frame of unnormalised predictions.
    """

    if self._target_scaler is None:
      raise ValueError('Scalers have not been set!')

    column_names = predictions.columns

    df_list = []
    for identifier, sliced in predictions.groupby('identifier'):
      sliced_copy = sliced.copy()
      target_scaler = self._target_scaler[identifier]

      for col in column_names:
        if col not in {'forecast_time', 'identifier'}:
          sliced_copy[col] = target_scaler.inverse_transform(sliced_copy[col])
      df_list.append(sliced_copy)

    output = pd.concat(df_list, axis=0)

    return output
```

**5) get_fixed_params** 

: returns fixed model parameters for experiments 

```python
# Default params
  def get_fixed_params(self):
    """Returns fixed model parameters for experiments."""

    fixed_params = {
        'total_time_steps': 8 * 24,
        'num_encoder_steps': 7 * 24, #168
        'num_epochs': 100,
        'early_stopping_patience': 5,
        'multiprocessing_workers': 5
    }

    return fixed_params
```

- time_steps : **Total number** of input time steps per forecast date (i.e. Width of Temporal fusion decoder N)

  168 시간 관측 후 다음 24시간을 맞추므로 ! 

- num_encoder_steps: Size of LSTM encoder -- i.e. number of past time step before forecast date to use

**그 외**

**get_default_model_params**

```python
def get_default_model_params(self):
    """Returns default optimised model parameters."""

    model_params = {
        'dropout_rate': 0.1,
        'hidden_layer_size': 160,
        'learning_rate': 0.001,
        'minibatch_size': 64,
        'max_gradient_norm': 0.01,
        'num_heads': 4,
        'stack_size': 1
    }

    return model_params

```

**get_num_samples_for_calibration**

```python
def get_num_samples_for_calibration(self):
    """Gets the default number of training and validation samples.
    Use to sub-sample the data for network calibration and a value of -1 uses
    all available samples.
    Returns:
      Tuple of (training samples, validation samples)
    """
    return 450000, 50000
```

