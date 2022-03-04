console.log('This example is different!');
console.log('The result is displayed in the Command Line Interface');

var http = require('http');
var date = require('./mymodule');

http.createServer(function (req, res) {
    res.writeHead(200, {'Content-Type': 'text/html'});
    res.write('date: ' + date.myDateTime());
    res.end();
}).listen(8080);

