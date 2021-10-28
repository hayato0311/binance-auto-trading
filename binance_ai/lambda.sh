mkdir -p ./lambda
rm -r ./lambda
mkdir -p ./lambda/

pip install -r requirements.txt -t ./lambda/

cp ./*.py lambda

chmod -R 755 ./lambda

touch lambda.zip
rm lambda.zip

cd lambda
zip -r ../lambda.zip .
