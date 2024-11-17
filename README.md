# Multilayer-Perceptron

## 単層パーセプトロンと多層パーセプトロンについて
パーセプトロンとは、複数の入力信号を受取り、一つの出力信号を送るもので、ニューロンやノードとも呼ばれる。
例えば、２つのニューロンから信号を受け取る単層のパーセプトロンについて考えると、これは論理回路で説明することができる。２つのニューロンを用いることでANDやOR、NANDといった回路を再現可能で、これは領域の線形的な分割を意味している。
しかし、XORといったもう少し複雑で、非線形的な回路の表現をしようとすると、単層パーセプトロンでは難しい。先ほどの回路に加え、NANDとORの出力をANDで結合した層を加えることでXORゲートを表現可能で、線形的な表現から非線形的な表現を実現しており、これはまさに多層パーセプトロンの基礎そのものである。

