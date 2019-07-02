xquery version "1.0-ml";

(:
  マルチパーセプトロンの入力層と隠れ層を構築する。出力層は含まない。
  input-variable : 入力層の入力数
  num_hidden_layers : 隠れ層のレイヤー数
  dense-map : アクティベーションのアルゴリズムや隠れ層の出力数などのパラメータを保持するmap
:)
declare function local:mlp($input-variable as cntk:variable, $num_hidden_layers as xs:integer, $dense-map as map:map){
  if($num_hidden_layers = 0) then cntk:dense-layer($input-variable, $dense-map)
  else cntk:dense-layer(local:mlp($input-variable, ($num_hidden_layers - 1), $dense-map), $dense-map)
};

(: マルチパーセプトロンの構成 :)
let $input-dims := 4 (:入力数:)
let $num-classes := 3 (:分類数:)
let $num_hidden_layers := 1 (:隠れ層の数:)
let $hidden_layers_dim := 50 (:隠れ層の出力数:)

let $hidden_activation := "relu"

(:学習データの準備:)
let $input-variable := cntk:input-variable(cntk:shape(( 1, $input-dims)), "float")
let $train-data :=
  for $x in xdmp:directory("/iris/train/", "infinity")
    let $train := fn:concat($x/iris/sepal_length/text(),",", $x/iris/sepal_width/text(), ",", $x/iris/petal_length/text(), ",", $x/iris/petal_width/text())
    (:return $train:)
    return ($x/iris/sepal_length/text(), $x/iris/sepal_width/text(), $x/iris/petal_length/text(),  $x/iris/petal_width/text())

let $input-value := cntk:batch(cntk:shape((1, $input-dims)), json:to-array($train-data))

(:学習用ラベルの準備:)
let $label-variable := cntk:input-variable(cntk:shape(( 1, $num-classes)), "float")
let $labels :=
for $x in xdmp:directory("/iris/train/", "infinity")
  let $data := $x
  let $_label := $x/iris/species/text()
  return if ($_label = "setosa") then (1,0,0)
  else if ($_label = "virginica")  then (0,1,0)
  else (0,0,1)
  
let $label-value := cntk:batch(cntk:shape((1,$num-classes)), json:to-array(($labels)))

(:::::::::::::::::::::::::::::::
マルチパーセプトロンを組み立てる
:::::::::::::::::::::::::::::::)
(:入力層と隠れ層を組み立てる:)
let $_model-map := map:map()
let $_ := map:put($_model-map, "activation", $hidden_activation)
let $_ := map:put($_model-map, "output-shape", cntk:shape((1, $hidden_layers_dim)))
let $_model := local:mlp($input-variable, ($num_hidden_layers), $_model-map)

let $_output-map := map:map()
let $_ := map:put($_output-map, "activation",$hidden_activation)
let $_ := map:put($_output-map, "output-shape", cntk:shape((1, $num-classes)))
let $_model := cntk:dense-layer($_model, $_output-map)

(:出力層を組み立てる。出力層のアクティベーションはsoftmax:)
let $model := cntk:softmax($_model)

(:予測結果と正解の交差エントロピー誤差を求める:)
let $loss := cntk:cross-entropy-with-softmax(
     $model, 
     $label-variable, 
     cntk:axis(-1))
     
(:予測結果の評価:)     
let $err := cntk:classification-error($model, $label-variable, 1, cntk:axis(-1))

(:訓練機の構築:)
let $parameter:=cntk:function-parameters($model)  
let $learner := 
     cntk:sgd-learner(($parameter), 
     cntk:learning-rate-schedule-from-constant(0.1))
let $trainer := cntk:trainer($model, ($learner), $loss, $err)

(:学習データと正解データからミニバッチ用データを組み立てる:)
let $input-pair := json:to-array(($input-variable, $input-value))
let $labels-pair := json:to-array(($label-variable, $label-value))
let $minibatch := json:to-array(($input-pair, $labels-pair))

(:ミニバッチ実行:)
let $train-result :=
    for $i in (1 to 50)
      let $__ := cntk:train-minibatch($trainer, $minibatch, fn:false())
      
      let $loss := cntk:previous-minibatch-loss-average($trainer)
      let $evaluation := (1.0 - cntk:previous-minibatch-evaluation-average($trainer)) 
      
      return json:to-array(($loss, $evaluation))

(:モデルの保存:)
let $model_doc := cntk:function-save($model)
let $__ := xdmp:document-insert("/model/iris_model", $model_doc)

return $train-result

