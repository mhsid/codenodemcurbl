//Include da lib de Wifi do ESP8266
#include <ESP8266WiFi.h>

//Definir o SSID da rede WiFi
const char* ssid = "rede";
//Definir a senha da rede WiFi
const char* password = "senha";

//Colocar a API Key para escrita neste campo
//Ela é fornecida no canal que foi criado na aba API Keys
String apiKey = "L174FD5B93LP7NIT";
const char* server = "api.thingspeak.com";

WiFiClient client;

void setup() {
  //Configuração da UART
  Serial.begin(9600);
  //Inicia o WiFi
  WiFi.begin(ssid, password);

  //Espera a conexão no router
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  void loop() {

    //Inicia um client TCP para o envio dos dados
    if (client.connect(server, 80)) {
      String postStr = apiKey;
      postStr += "&amp;field1=";
      postStr += String(variavel);
      postStr += "\r\n\r\n";

      client.print("POST /update HTTP/1.1\n");
      client.print("Host: api.thingspeak.com\n");
      client.print("Connection: close\n");
      client.print("X-THINGSPEAKAPIKEY: " + apiKey + "\n");
      client.print("Content-Type: application/x-www-form-urlencoded\n");
      client.print("Content-Length: ");
      client.print(postStr.length());
      client.print("\n\n");
      client.print(postStr);

    }
    client.stop();
  }
