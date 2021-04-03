## data_formatters

### base.py

**필요한 모듈 가져오기** 

```python
import abc
import enum
```

1. abc 모듈 - 추상클래스 

​       출처 : [파이썬 코딩 도장 - 추상클래스](https://www.youtube.com/watch?v=YiuTxiTi7aE)

​       예시 ) 

```python
from abc import *

class StudentBase(metaclass=ABCMeta):
  @abstractmethod
  def study(self):
    pass
  
  @abstractmethod
  def go_to_school(self):
    pass
  
  class Student(StudentBase):
    def study(self):
      print('공부하기')
     
    def go_to_school(self):
      print('학교가기')
 
james=Student()
james.study()
james.go_to_school()
```

2. enum 모듈 - 열거형 정의

   데이터를 "이름=값" 형태로 저장할 수 있음 

   출처 : <파이썬 라이브러리 레시피>

   ```python
   import enum
   class Dynasty(enum.Enum):
     GOGURYEO=1
     BAEKJE=2
     SILLA=3
   
   dynasty=Dynasty.SILLA
   ```

   <Dynasty.SILLA: 3>

**데이터 타입, 인풋 타입 정의하기** 

데이터 타입

```python
class DataTypes(enum.IntEnum):
  """Defines numerical types of each column."""
  REAL_VALUED = 0
  CATEGORICAL = 1
  DATE = 2
```

TFT 에서 시계열 데이터 타입은 실수형/ 카테고리형/날짜형 이렇게 세 개이다.

- 실수형
- 카테고리형
- 날짜형

인풋 타입

```python
class InputTypes(enum.IntEnum):
  """Defines input types of each column."""
  TARGET = 0
  OBSERVED_INPUT = 1
  KNOWN_INPUT = 2
  STATIC_INPUT = 3
  ID = 4  # Single column used as an entity identifier
  TIME = 5  # Single column exclusively used as a time index
```

TFT 에서 인풋 타입은 타겟 (y 값) / 측정 값/측정하지 않아도 알 수 있는 기본 값/메타데이터/ID/시간(t) 이렇게 여섯 개이다. 

- y 값 (target)
- 측정 값
- 측정 X, 기본 값
- 메타 데이터
- ID
- 시간 (t)

**추상 클래스 정의하기** 

```python
class GenericDataFormatter(abc.ABC):
  """Abstract base class for all data formatters.
  User can implement the abstract methods below to perform dataset-specific
  manipulations.
  """

  @abc.abstractmethod
  def set_scalers(self, df):
    """Calibrates scalers using the data supplied."""
    raise NotImplementedError()

  @abc.abstractmethod
  def transform_inputs(self, df):
    """Performs feature transformation."""
    raise NotImplementedError()

  @abc.abstractmethod
  def format_predictions(self, df):
    """Reverts any normalisation to give predictions in original scale."""
    raise NotImplementedError()

  @abc.abstractmethod
  def split_data(self, df):
    """Performs the default train, validation and test splits."""
    raise NotImplementedError()
```

정의된 추상 클래스를 통해서 총 네 개의 메소드를 사용함을 알 수 있다. 각 메소드의 구체적 기능은 상속 받은 클래스에 정의된 사항들을 보면 알 수 있을 것 같다.

- set_scalers
- transform_inputs
- format_predictions
- split_data

**클래스 내 프로퍼티 사용하기**  

이후 정의된 메소드도 쭉 살펴보면 @property 로 프로퍼티를 사용한 메소드들도 꽤 보입니다. 

```python
@property
  @abc.abstractmethod
  def _column_definition(self):
    """Defines order, input type and data type of each column."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_fixed_params(self):
    """Defines the fixed parameters used by the model for training.
    Requires the following keys:
      'total_time_steps': Defines the total number of time steps used by TFT
      'num_encoder_steps': Determines length of LSTM encoder (i.e. history)
      'num_epochs': Maximum number of epochs for training
      'early_stopping_patience': Early stopping param for keras
      'multiprocessing_workers': # of cpus for data processing
    Returns:
      A dictionary of fixed parameters, e.g.:
      fixed_params = {
          'total_time_steps': 252 + 5,
          'num_encoder_steps': 252,
          'num_epochs': 100,
          'early_stopping_patience': 5,
          'multiprocessing_workers': 5,
      }
    """
    raise NotImplementedError

  # Shared functions across data-formatters
  @property
  def num_classes_per_cat_input(self):
    """Returns number of categories per relevant input.
    This is seqeuently required for keras embedding layers.
    """
    return self._num_classes_per_cat_input

  def get_num_samples_for_calibration(self):
    """Gets the default number of training and validation samples.
    Use to sub-sample the data for network calibration and a value of -1 uses
    all available samples.
    Returns:
      Tuple of (training samples, validation samples)
    """
    return -1, -1

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

    identifier = [tup for tup in column_definition if tup[2] == InputTypes.ID]
    time = [tup for tup in column_definition if tup[2] == InputTypes.TIME]
    real_inputs = [
        tup for tup in column_definition if tup[1] == DataTypes.REAL_VALUED and
        tup[2] not in {InputTypes.ID, InputTypes.TIME}
    ]
    categorical_inputs = [
        tup for tup in column_definition if tup[1] == DataTypes.CATEGORICAL and
        tup[2] not in {InputTypes.ID, InputTypes.TIME}
    ]

    return identifier + time + real_inputs + categorical_inputs

  def _get_input_columns(self):
    """Returns names of all input columns."""
    return [
        tup[0]
        for tup in self.get_column_definition()
        if tup[2] not in {InputTypes.ID, InputTypes.TIME}
    ]

  def _get_tft_input_indices(self):
    """Returns the relevant indexes and input sizes required by TFT."""

    # Functions
    def _extract_tuples_from_data_type(data_type, defn):
      return [
          tup for tup in defn if tup[1] == data_type and
          tup[2] not in {InputTypes.ID, InputTypes.TIME}
      ]

    def _get_locations(input_types, defn):
      return [i for i, tup in enumerate(defn) if tup[2] in input_types]

    # Start extraction
    column_definition = [
        tup for tup in self.get_column_definition()
        if tup[2] not in {InputTypes.ID, InputTypes.TIME}
    ]

    categorical_inputs = _extract_tuples_from_data_type(DataTypes.CATEGORICAL,
                                                        column_definition)
    real_inputs = _extract_tuples_from_data_type(DataTypes.REAL_VALUED,
                                                 column_definition)

    locations = {
        'input_size':
            len(self._get_input_columns()),
        'output_size':
            len(_get_locations({InputTypes.TARGET}, column_definition)),
        'category_counts':
            self.num_classes_per_cat_input,
        'input_obs_loc':
            _get_locations({InputTypes.TARGET}, column_definition),
        'static_input_loc':
            _get_locations({InputTypes.STATIC_INPUT}, column_definition),
        'known_regular_inputs':
            _get_locations({InputTypes.STATIC_INPUT, InputTypes.KNOWN_INPUT},
                           real_inputs),
        'known_categorical_inputs':
            _get_locations({InputTypes.STATIC_INPUT, InputTypes.KNOWN_INPUT},
                           categorical_inputs),
    }

    return locations

  def get_experiment_params(self):
    """Returns fixed model parameters for experiments."""

    required_keys = [
        'total_time_steps', 'num_encoder_steps', 'num_epochs',
        'early_stopping_patience', 'multiprocessing_workers'
    ]

    fixed_params = self.get_fixed_params()

    for k in required_keys:
      if k not in fixed_params:
        raise ValueError('Field {}'.format(k) +
                         ' missing from fixed parameter definitions!')

    fixed_params['column_definition'] = self.get_column_definition()

    fixed_params.update(self._get_tft_input_indices())

    return fixed_params
```

1. 프로퍼티 사용하기 

출처 : [파이썬 코딩 도장 - 추상클래스](https://www.youtube.com/watch?v=YiuTxiTi7aE)

예시 ) 

```python
class Person:
  def __init__(self):
    self.__age=0
    
  @property            #getter
  def age(self):
    return self.__age
  
  @age.setter         #setter
  def age(self,value):
    self.__age=value
```

누가 파이썬이 쉽다고 했나요. 정말 참 양파같은 친구라는 생각이 듭니다. 🧅

**TFT / data_formatters 에서 정의된 GenericDataFormatter 클래스 전체 코드** 

```python
class GenericDataFormatter(abc.ABC):
  """Abstract base class for all data formatters.
  User can implement the abstract methods below to perform dataset-specific
  manipulations.
  """

  @abc.abstractmethod
  def set_scalers(self, df):
    """Calibrates scalers using the data supplied."""
    raise NotImplementedError()

  @abc.abstractmethod
  def transform_inputs(self, df):
    """Performs feature transformation."""
    raise NotImplementedError()

  @abc.abstractmethod
  def format_predictions(self, df):
    """Reverts any normalisation to give predictions in original scale."""
    raise NotImplementedError()

  @abc.abstractmethod
  def split_data(self, df):
    """Performs the default train, validation and test splits."""
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def _column_definition(self):
    """Defines order, input type and data type of each column."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_fixed_params(self):
    """Defines the fixed parameters used by the model for training.
    Requires the following keys:
      'total_time_steps': Defines the total number of time steps used by TFT
      'num_encoder_steps': Determines length of LSTM encoder (i.e. history)
      'num_epochs': Maximum number of epochs for training
      'early_stopping_patience': Early stopping param for keras
      'multiprocessing_workers': # of cpus for data processing
    Returns:
      A dictionary of fixed parameters, e.g.:
      fixed_params = {
          'total_time_steps': 252 + 5,
          'num_encoder_steps': 252,
          'num_epochs': 100,
          'early_stopping_patience': 5,
          'multiprocessing_workers': 5,
      }
    """
    raise NotImplementedError

  # Shared functions across data-formatters
  @property
  def num_classes_per_cat_input(self):
    """Returns number of categories per relevant input.
    This is seqeuently required for keras embedding layers.
    """
    return self._num_classes_per_cat_input

  def get_num_samples_for_calibration(self):
    """Gets the default number of training and validation samples.
    Use to sub-sample the data for network calibration and a value of -1 uses
    all available samples.
    Returns:
      Tuple of (training samples, validation samples)
    """
    return -1, -1

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

    identifier = [tup for tup in column_definition if tup[2] == InputTypes.ID]
    time = [tup for tup in column_definition if tup[2] == InputTypes.TIME]
    real_inputs = [
        tup for tup in column_definition if tup[1] == DataTypes.REAL_VALUED and
        tup[2] not in {InputTypes.ID, InputTypes.TIME}
    ]
    categorical_inputs = [
        tup for tup in column_definition if tup[1] == DataTypes.CATEGORICAL and
        tup[2] not in {InputTypes.ID, InputTypes.TIME}
    ]

    return identifier + time + real_inputs + categorical_inputs

  def _get_input_columns(self):
    """Returns names of all input columns."""
    return [
        tup[0]
        for tup in self.get_column_definition()
        if tup[2] not in {InputTypes.ID, InputTypes.TIME}
    ]

  def _get_tft_input_indices(self):
    """Returns the relevant indexes and input sizes required by TFT."""

    # Functions
    def _extract_tuples_from_data_type(data_type, defn):
      return [
          tup for tup in defn if tup[1] == data_type and
          tup[2] not in {InputTypes.ID, InputTypes.TIME}
      ]

    def _get_locations(input_types, defn):
      return [i for i, tup in enumerate(defn) if tup[2] in input_types]

    # Start extraction
    column_definition = [
        tup for tup in self.get_column_definition()
        if tup[2] not in {InputTypes.ID, InputTypes.TIME}
    ]

    categorical_inputs = _extract_tuples_from_data_type(DataTypes.CATEGORICAL,
                                                        column_definition)
    real_inputs = _extract_tuples_from_data_type(DataTypes.REAL_VALUED,
                                                 column_definition)

    locations = {
        'input_size':
            len(self._get_input_columns()),
        'output_size':
            len(_get_locations({InputTypes.TARGET}, column_definition)),
        'category_counts':
            self.num_classes_per_cat_input,
        'input_obs_loc':
            _get_locations({InputTypes.TARGET}, column_definition),
        'static_input_loc':
            _get_locations({InputTypes.STATIC_INPUT}, column_definition),
        'known_regular_inputs':
            _get_locations({InputTypes.STATIC_INPUT, InputTypes.KNOWN_INPUT},
                           real_inputs),
        'known_categorical_inputs':
            _get_locations({InputTypes.STATIC_INPUT, InputTypes.KNOWN_INPUT},
                           categorical_inputs),
    }

    return locations

  def get_experiment_params(self):
    """Returns fixed model parameters for experiments."""

    required_keys = [
        'total_time_steps', 'num_encoder_steps', 'num_epochs',
        'early_stopping_patience', 'multiprocessing_workers'
    ]

    fixed_params = self.get_fixed_params()

    for k in required_keys:
      if k not in fixed_params:
        raise ValueError('Field {}'.format(k) +
                         ' missing from fixed parameter definitions!')

    fixed_params['column_definition'] = self.get_column_definition()

    fixed_params.update(self._get_tft_input_indices())

    return fixed_params
```

정리해보면

**@abc.abstractmethod 를 사용한 추상 클래스 메소드 (setter 로 값 가져오기 x)**

- set_scalers : calibrates scalers using the data supplied
- transform_inputs : perform feature transformation
- format_predictions : reverts any normalization to give predictions in original scale 
- split_data : performs the default train, validation and test splits
- get_fixed_params : defines the fixed parameters used by model for training 

**@abc.abstractmethod 를 사용한 추상 클래스 메소드 (+ setter 로 값 가져오기)**

- _column_definition : defines order, input type and data type of each column 

**setter 로 프로퍼티 속성을 사용한 메소드 (추상 클래스 메소드 x)**

- num_classes_per_cat_input : returns number of categories per relevant input 

**그 외 data_formatters 에 정의된 메소드 (추상 메소드 아님, setter 아님)**

- get_num_samples_for_calibration : gets the default number of training and validation samples 

- get_column_definition : returns formatted column definition in order expected by the TFT 

- _check_single_column : Ensure only one ID and time column exists 

- _get_input_columns : returns names of all input columns 

- _get_tft_input_indices : returns the relevant indexes and input sizes required by TFT

- get_experiment_params : returns fixed model parameters for experiments

  

