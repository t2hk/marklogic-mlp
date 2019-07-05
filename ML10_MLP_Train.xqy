xquery version "1.0-ml";

(:
  マルチパーセプトロンの入力層と隠れ層を構築する。出力層は含まない。
  input_variable : 入力層の入力数
  num_hidden_layers : 隠れ層のレイヤー数
  dense_map : アクティベーションのアルゴリズムや隠れ層の出力数などのパラメータを保持するmap
:)
declare function local:mlp($input_variable as cntk:variable, $num_hidden_layers as xs:integer, $dense_map as map:map){
  if($num_hidden_layers = 0) then cntk:dense-layer($input_variable, $dense_map)
  else cntk:dense-layer(local:mlp($input_variable, ($num_hidden_layers - 1), $dense_map), $dense_map)
};

(: マルチパーセプトロンの構成 :)
let $input_dims := 4 (:入力数:)
let $num_classes := 3 (:分類数:)
let $num_hidden_layers := 2 (:隠れ層の数:)
let $hidden_layers_dim := 10 (:隠れ層の出力数:)
let $hidden_activation := "relu"

(:ミニバッチの実行回数:)
let $minibatch_size := 100

(:学習データの準備:)
let $input_variable := cntk:input-variable(cntk:shape(( 1, $input_dims)), "float")
let $train_data :=
  for $x in xdmp:directory("/iris/train/", "infinity")
    return ($x/iris/sepal_length/text(), $x/iris/sepal_width/text(), $x/iris/petal_length/text(),  $x/iris/petal_width/text())

let $input_value := cntk:batch(cntk:shape((1, $input_dims)), json:to-array($train_data))

(:学習用ラベルの準備:)
let $label_variable := cntk:input-variable(cntk:shape(( 1, $num_classes)), "float")
let $labels :=
for $x in xdmp:directory("/iris/train/", "infinity")
  let $_label := $x/iris/species/text()
  return if ($_label = "setosa") then (1,0,0)
  else if ($_label = "virginica")  then (0,1,0)
  else (0,0,1)
  
let $label_value := cntk:batch(cntk:shape((1,$num_classes)), json:to-array(($labels)))

(:::::::::::::::::::::::::::::::
マルチパーセプトロンを組み立てる
:::::::::::::::::::::::::::::::)
(:入力層と隠れ層を組み立てる:)
let $_model_map := map:map()
let $_ := map:put($_model_map, "activation", $hidden_activation)
let $_ := map:put($_model_map, "output-shape", cntk:shape((1, $hidden_layers_dim)))
let $_model := local:mlp($input_variable, ($num_hidden_layers), $_model_map)

(:出力層を組み立てる:)
let $_output_map := map:map()
let $_ := map:put($_output_map, "activation", $hidden_activation)
let $_ := map:put($_output_map, "output-shape", cntk:shape((1, $num_classes)))
let $model := cntk:dense-layer($_model, $_output_map)

(:予測結果と正解の交差エントロピー誤差を求める:)
let $loss := cntk:cross-entropy-with-softmax(
     $model, 
     $label_variable, 
     cntk:axis(-1))
     
(:分類エラーの評価:)     
let $err := cntk:classification-error($model, $label_variable, 1, cntk:axis(-1))

(:訓練器の構築:)
let $parameter:=cntk:function-parameters($model)

let $learner := 
     cntk:sgd-learner(($parameter), 
     cntk:learning-rate-schedule-from-constant(0.1))
let $trainer := cntk:trainer($model, ($learner), $loss, $err)

(:学習データと正解データからミニバッチ用データを組み立てる:)
let $input_pair := json:to-array(($input_variable, $input_value))
let $labels_pair := json:to-array(($label_variable, $label_value))
let $minibatch := json:to-array(($input_pair, $labels_pair))

(:ミニバッチ実行:)
let $train_result :=
    for $i in (1 to $minibatch_size)
      let $__ := cntk:train-minibatch($trainer, $minibatch, fn:false())
      
      let $loss := cntk:previous-minibatch-loss-average($trainer)
      let $evaluation := (1.0 - cntk:previous-minibatch-evaluation-average($trainer)) 
      
      return json:to-array(($loss, $evaluation))

(:モデルの保存:)
let $model_doc := cntk:function-save($model)
let $__ := xdmp:document-insert("/model/iris_model", $model_doc)

return $train_result

