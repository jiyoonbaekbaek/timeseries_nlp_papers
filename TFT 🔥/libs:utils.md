## libs : contains the main libraries, including classes to manage hyperparameter optimization 

### utils.py : Generic helper functions used across codebase

helper function 이란 ? `A helper function is a function that performs part of the computation of another function` - from google 

**필요한 모듈 임포트** 

```python
import os
import pathlib

import numpy as np
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
```

- tensorflow.python.tools.inspect_checkpoint 설명 

   [Several methods Tensorflow output saved in checkpoint variable](https://www.programmersought.com/article/79712700244/)

**helper function 1) get_single_col_by_input_type** 

```python
def get_single_col_by_input_type(input_type, column_definition):
  """Returns name of single column.
  Args:
    input_type: Input type of column to extract
    column_definition: Column definition list for experiment
  """

  l = [tup[0] for tup in column_definition if tup[2] == input_type]

  if len(l) != 1:
    raise ValueError('Invalid number of columns for {}'.format(input_type))

  return l[0]
```

i.e.    column_definition 을 

[
      ('id', DataTypes.REAL_VALUED, InputTypes.ID),
      ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.TIME),
      ('power_usage', DataTypes.REAL_VALUED, InputTypes.TARGET),
      ('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('categorical_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
  ]

이라고 하면,

identifier=[('id', 'DataTypes.REAL_VALUED', 'InputTypes.ID')]

time = [ ('hours_from_start', 'DataTypes.REAL_VALUED', 'InputTypes.TIME')]

real_inputs=[('power_usage', 'DataTypes.REAL_VALUED', 'InputTypes.TARGET'),('day_of_week', 'DataTypes.REAL_VALUED', 'InputTypes.KNOWN_INPUT'),('hours_from_start', 'DataTypes.REAL_VALUED', 'InputTypes.KNOWN_INPUT')]

categorical_inputs=[('categorical_id', 'DataTypes.CATEGORICAL', 'InputTypes.STATIC_INPUT')]

```python
column_definitions = self.get_column_definition()
id_col = utils.get_single_col_by_input_type(InputTypes.ID,column_definitions)
```

이런 식으로 helper function 이 쓰이는 것을 알 수 있다.

get_column_definition 메소드는 

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


```

이므로 column_definition 에는 모든 칼럼 정보가 들어간다. 이런 식으로

```python
[('id', DataTypes.REAL_VALUED, InputTypes.ID), ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.TIME), ('power_usage', DataTypes.REAL_VALUED, InputTypes.TARGET), ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT), ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT), ('categorical_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT)]
```

따라서 

```python
id_col = utils.get_single_col_by_input_type(InputTypes.ID,column_definitions)
```

에는 ['id'] 가 이렇게 담기게 된다.

**helper function 2) extract_cols_from_data_type** 

```python
def extract_cols_from_data_type(data_type, column_definition,
                                excluded_input_types):
  """Extracts the names of columns that correspond to a define data_type.
  Args:
    data_type: DataType of columns to extract.
    column_definition: Column definition to use.
    excluded_input_types: Set of input types to exclude
  Returns:
    List of names for columns with data type specified.
  """
  return [
      tup[0]
      for tup in column_definition
      if tup[1] == data_type and tup[2] not in excluded_input_types
  ]

```

해당 helper 메소드는

```python
categorical_inputs = utils.extract_cols_from_data_type(DataTypes.CATEGORICAL, column_definitions,{InputTypes.ID, InputTypes.TIME})
```

이런 식으로 쓰인다. 

column_definition 이 아까와 동일하게

```python
[('id', DataTypes.REAL_VALUED, InputTypes.ID), ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.TIME), ('power_usage', DataTypes.REAL_VALUED, InputTypes.TARGET), ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT), ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT), ('categorical_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT)]
```

이라면, 따라서 datatype 이 categorical 이면서 inputtype 이 ID, TIME 이 아닌 'categorical_id' 가 return 된다.

한마디로  get_single_col_by_input_type 헬프메소드는 인자로 id 나 time 이 들어가서 해당 유니크한 열을 뽑아낼 때 쓰이고 (애초에 id,time feature 은 모든 데이터에 한 개씩 밖에 없으므로) , extract_cols_from_data_type 헬프 메소드는 id 나 time 유니크한 칼럼을 제외하고 해당 데이터 타입을 갖는 모든 열들을 뽑아낼 때 쓰인다. 

**helper function 3) tensorflow_quantile_loss**

```python
# Loss functions.
def tensorflow_quantile_loss(y, y_pred, quantile):
  """Computes quantile loss for tensorflow.
  Standard quantile loss as defined in the "Training Procedure" section of
  the main TFT paper
  Args:
    y: Targets
    y_pred: Predictions
    quantile: Quantile to use for loss calculations (between 0 & 1)
  Returns:
    Tensor for quantile loss.
  """

  # Checks quantile
  if quantile < 0 or quantile > 1:
    raise ValueError(
        'Illegal quantile value={}! Values should be between 0 and 1.'.format(
            quantile))

  prediction_underflow = y - y_pred #오차
  q_loss = quantile * tf.maximum(prediction_underflow, 0.) + (
      1. - quantile) * tf.maximum(-prediction_underflow, 0.)

  return tf.reduce_sum(q_loss, axis=-1)
```

- tf.reduce_sum

   [tf.reduce_sum()의 의미](https://m.blog.naver.com/PostView.nhn?blogId=kmkim1222&logNo=220992490164&proxyReferer=https:%2F%2Fwww.google.com%2F)

- quantile loss

  ![스크린샷 2021-04-03 오후 12 51 58](https://user-images.githubusercontent.com/67775336/113467054-6f0ddd80-947b-11eb-9307-78755438d502.png)



**helper function 4) numpy_normalised_quantile_loss** 

```python
def numpy_normalised_quantile_loss(y, y_pred, quantile):
  """Computes normalised quantile loss for numpy arrays.
  Uses the q-Risk metric as defined in the "Training Procedure" section of the
  main TFT paper.
  Args:
    y: Targets
    y_pred: Predictions
    quantile: Quantile to use for loss calculations (between 0 & 1)
  Returns:
    Float for normalised quantile loss.
  """
  prediction_underflow = y - y_pred
  weighted_errors = quantile * np.maximum(prediction_underflow, 0.) \
      + (1. - quantile) * np.maximum(-prediction_underflow, 0.)

  quantile_loss = weighted_errors.mean()
  normaliser = y.abs().mean()

  return 2 * quantile_loss / normaliser
```

- Out of sample test 용으로 조금 변형된 quantile loss

  ![스크린샷 2021-04-03 오후 12 55 53](https://user-images.githubusercontent.com/67775336/113467129-f0fe0680-947b-11eb-88e0-e6f965a06a54.png)



**helper funtion 5) print_weights_in_checkpoint**

```python
def print_weights_in_checkpoint(model_folder, cp_name):
  """Prints all weights in Tensorflow checkpoint.
  Args:
    model_folder: Folder containing checkpoint
    cp_name: Name of checkpoint
  Returns:
  """
  load_path = os.path.join(model_folder, '{0}.ckpt'.format(cp_name))

  print_tensors_in_checkpoint_file(
      file_name=load_path,
      tensor_name='',
      all_tensors=True,
      all_tensor_names=True)
```

- os.path.join : 경로를 병합하여 새 경로 생성 

  [파이썬에서 파일과 디렉토리 경로 다루기](http://pythonstudy.xyz/python/article/507-%ED%8C%8C%EC%9D%BC%EA%B3%BC-%EB%94%94%EB%A0%89%ED%86%A0%EB%A6%AC)

  ```python
  os.path.join('C:\Tmp', 'a', 'b')
  # "C:\Tmp\a\b"
  ```

그러면 loaded_path 는 model_folder/cp_name 이런 식으로 나올꺼다. Ex) /tmp/train1.ckpt

- 텐서플로우에서 checkpoint 란 학습된 모델의 variable 값을 저장하는 파일 

[tensorflow 로 checkpoint 파일 이용하기](http://jaynewho.com/post/8)

그 외에 helper function 중에 cpu,gpu 옵션이나 checkpoint 저장 등등에 관련된 부분은 살펴보지 않았다. 

