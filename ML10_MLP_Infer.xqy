xquery version "1.0-ml";

let $input_dims := 4
let $num_classes := 3

let $label_name := json:to-array(("setosa", "virginica", "versicolor"))

(:学習データの準備:)
let $model :=
    cntk:function(fn:doc("/model/iris_model")/binary())

let $input_variable := cntk:function-arguments($model)
let $output_variable := cntk:function-output($model)

(:::::::::::::::::::::::::::::::::::::::::::::::::::
推論データを読み込み、入力データと正解ラベルを組み立てる
:::::::::::::::::::::::::::::::::::::::::::::::::::)
let $test_data_docs :=
  for $doc in xdmp:directory("/iris/test/", "infinity")
    return $doc

(:入力データの組み立て:)
let $test_data :=
  for $x in $test_data_docs
    let $_test_data := ($x/iris/sepal_length/text(), $x/iris/sepal_width/text(), $x/iris/petal_length/text(),  $x/iris/petal_width/text())
    return $_test_data
    
(:正解ラベルの組み立て:)
let $test_labels :=
  for $x in $test_data_docs
    let $_label := $x/iris/species/text()
    let $_one_hot_label := 
      if ($_label = "setosa") then ("setosa", 1,0,0)
      else if ($_label = "virginica")  then ("virginica", 0,1,0)
      else if ($_label = "versicolor") then ("versicolor", 0,0,1)
      else ("unknown", 0,0,0)

    return json:to-array($_one_hot_label)
    
(: 推論実行 :)
let $test_value := cntk:batch(cntk:shape((1, $input_dims)), json:to-array($test_data))
let $test_pair := json:to-array(($input_variable, $test_value))
let $output_value := cntk:evaluate(
     $model, 
     $test_pair, 
     ($output_variable))
     
(:推論結果と正解データを組み合わせる:)     
let $output_sequence := json:array-values(
     cntk:value-to-array($output_variable, 
     $output_value), 
     fn:true())
let $result_array := cntk:value-to-array($output_variable, $output_value)

(::::::::::::::::::::::::
結果の表示
::::::::::::::::::::::::)
let $size := json:array-size($result_array)

(:入力値、正解ラベル、推論結果、推論ラベルを出力する:)
let $result_value :=
for $pos in 1 to $size
   let $data_pos := $pos * 4 - 3 
   let $input_value := json:to-array(($test_data[$data_pos], $test_data[($data_pos + 1)], $test_data[($data_pos + 2)], $test_data[($data_pos + 3)]))
   let $one_hot_label := json:to-array($test_labels[$pos])
  
  let $infer := json:to-array($result_array[$pos])
  let $max_pos := fn:index-of($result_array[$pos], fn:max($result_array[$pos]))
   
  let $infer_label := 
    if (fn:count($max_pos) = 1) then $label_name[$max_pos]
    else "unknown"
  
  (:推論結果が正しいかチェックする:)
  let $is_valid :=
    if ($one_hot_label[1] = $infer_label) then fn:true()
    else fn:false()    
      
   return json:to-array(($is_valid, $input_value, $one_hot_label, $infer, $infer_label))

(:正解した件数をカウントする:)
let $valid_count :=
  for $res in $result_value
    return
      if ($res[1] = fn:true()) then 1
      else (0)

return (fn:concat("正解率：", fn:sum($valid_count),"/",$size), $result_value)