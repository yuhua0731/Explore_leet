#!/usr/bin/env node
var http = require("http");
var url = require("url");

http
  .createServer(function (req, res) {
    res.writeHead(200, { "Content-Type": "text/html" });
    var q = url.parse(req.url, true).query;
    res.end(q.year + " " + q.month);
  })
  .listen(8080);
