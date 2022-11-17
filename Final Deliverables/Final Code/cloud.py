from cloudant.client import Cloudant

client = Cloudant.iam('947534e7-5837-41fa-b216-7aaace1a2275-bluemix', '6b6UgDIZZDVt0BklyqhcQzjVEQZIpws6YOGgzjb2Tg8U',
                      connect=True)
my_database = client.create_database('my_database')