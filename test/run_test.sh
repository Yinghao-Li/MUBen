
i=3
RUN_TEST_1=123
RUN_TEST_2=234

if [ -z ${"RUN_TEST$i"+x} ];
then
  echo "skip test"
else
  python test.py \
    --x 123 \
    --y True
fi
