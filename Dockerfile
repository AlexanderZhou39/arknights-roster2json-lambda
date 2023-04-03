# FROM lambci/lambda:build-python3.8
FROM public.ecr.aws/lambda/python:3.8
RUN pip3 install --upgrade pip
COPY requirements.txt  .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

COPY . .

CMD [ "handler.main" ] 

