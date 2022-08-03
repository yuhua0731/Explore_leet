#!/usr/bin/env node
console.log("The result is displayed in the Command Line Interface");

var http = require("http");
var date = require("./mymodule");
var fs = require("fs");

// display current date
http
  .createServer(function (req, res) {
    res.writeHead(200, { "Content-Type": "text/html" });
    res.write("date: " + date.myDateTime());
    res.end();
  })
  .listen(8080);

// display content from file
http
  .createServer(function (req, res) {
    fs.readFile("demo.html", (err, data) => {
      res.writeHead(200, { "Content-Type": "text/html" });
      res.write(data);
      return res.end();
    });
  })
  .listen(8080);

