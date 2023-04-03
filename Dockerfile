# FROM lambci/lambda:build-python3.8
FROM public.ecr.aws/lambda/python:3.8
RUN pip3 install --upgrade pip
COPY requirements.txt  .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

ADD ./amazonlinux-2/lib/* /usr/lib64/

COPY ./amazonlinux-2/tesseract/share/tessdata /usr/tessdata

ENV TESSDATA_PREFIX /usr/tessdata

COPY . .

CMD [ "handler.main" ] 
