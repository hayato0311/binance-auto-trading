mkdir -p ./python
rm -r ./python
mkdir -p ./python

library="binance-connector"

pip install $library  -t ./python

chmod -R 755 ./python

zip -r $library.zip ./python

