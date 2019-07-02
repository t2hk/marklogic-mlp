xquery version "1.0-ml";

let $input-dims := 4
let $num-classes := 3

let $label-name := json:to-array(("setosa", "virginica", "versicolor"))

(:学習データの準備:)
let $model :=
    cntk:function(fn:doc("/model/iris_model")/binary())

let $input-variable := cntk:function-arguments($model)
let $output-variable := cntk:function-output($model)

(:::::::::::::::::::::::::::::::::::::::::::::::::::
推論データを読み込み、入力データと正解ラベルを組み立てる
:::::::::::::::::::::::::::::::::::::::::::::::::::)
let $test-data-docs :=
  for $doc in xdmp:directory("/iris/test/", "infinity")
    return $doc

(:入力データの組み立て:)
let $test-data :=
  for $x in $test-data-docs
    let $_test-data := ($x/iris/sepal_length/text(), $x/iris/sepal_width/text(), $x/iris/petal_length/text(),  $x/iris/petal_width/text())
    return $_test-data
    
(:正解ラベルの組み立て:)
let $test-labels :=
  for $x in $test-data-docs
    let $_label := $x/iris/species/text()
    let $_one-hot-label := 
      if ($_label = "setosa") then ("senosa", 1,0,0)
      else if ($_label = "virginica")  then ("virginica", 0,1,0)
      else if ($_label = "versicolor") then ("versicolor", 0,0,1)
      else ("unknown", 0,0,0)

    return json:to-array($_one-hot-label)
    
(: 推論実行 :)
let $test-value := cntk:batch(cntk:shape((1, $input-dims)), json:to-array($test-data))
let $test-pair := json:to-array(($input-variable, $test-value))
let $output-value := cntk:evaluate(
     $model, 
     $test-pair, 
     ($output-variable))
     
(:推論結果と正解データを組み合わせる:)     
let $output-sequence := json:array-values(
     cntk:value-to-array($output-variable, 
     $output-value), 
     fn:true())
let $result-array := cntk:value-to-array($output-variable, $output-value)

(::::::::::::::::::::::::
結果の表示
::::::::::::::::::::::::)
let $size := json:array-size($result-array)

(:入力値、正解ラベル、推論結果、推論ラベルを出力する:)
let $result-value :=
for $pos in 1 to $size
   let $data-pos := $pos * 4 - 3 
   let $input-value := json:to-array(($test-data[$data-pos], $test-data[($data-pos + 1)], $test-data[($data-pos + 2)], $test-data[($data-pos + 3)]))
   let $one-hot-label := json:to-array($test-labels[$pos])
  
  let $infer := json:to-array($result-array[$pos])
  let $max-pos := fn:index-of($result-array[$pos], fn:max($result-array[$pos]))
  let $infer-label := $label-name[$max-pos]
  
  (:推論結果が正しいかチェックする:)
  let $is_valid :=
    if ($one-hot-label[1] = $infer-label) then fn:true()
    else fn:false()    
      
   return json:to-array(($is_valid, $input-value, $one-hot-label, $infer, $infer-label))

(:正解した件数をカウントする:)
let $valid-count :=
  for $res in $result-value
    return
      if ($res[1] = fn:true()) then 1
      else (0)

return (fn:concat("正解率：", fn:sum($valid-count),"/",$size), $result-value)