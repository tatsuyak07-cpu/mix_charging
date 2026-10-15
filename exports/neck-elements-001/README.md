# ネック部品の局所細分化・要素形式の検証

0.5 / 0.25 / 0.125 mm、C3D8 / C3D8Iの6部品ケースと独立した梁試験です。
体積と階段形状を固定しました。C3D8Iの0.25→0.125 mmの反力・残留変形差は2%基準内ですが、局所塑性ひずみと充填層全体は未収束・未検証です。
summary.jsonとhistory.csvは元の節点平均の記録です。summary-area-weighted.jsonとhistory-area-weighted.csvおよび比較CSVは面積平均で再集計した値です。
部品の形状はmetadata.geometryで定義され、継承したconfigのlength_mm・porosity・target_mpa・mesh_mmは部品条件に適用していません。
圧力や空隙率を評価する周期充填層の計算ではなく、新規接触・損傷・温度依存性は含みません。
詳しくは docs/neck-element-verification.md を参照してください。
