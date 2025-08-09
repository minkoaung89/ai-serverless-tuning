for i in {1..15}; do
  aws lambda invoke --function-name lambda-tuning-demo --payload '{}' /dev/null
done
sleep 180

aws lambda invoke --function-name lambda-tuning-demo output.json
cat output.json