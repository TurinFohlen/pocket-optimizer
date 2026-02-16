from registry import registry
from orchestrator import Orchestrator, OptimizationConfig
from flask import Flask, request, jsonify

@registry.register(
    name='ui.web_demo',
    type_='ui',
    signature='run()'
)
class WebUI:
    def run(self, host='0.0.0.0', port=5000):
        app = Flask(__name__)
        
        @app.route('/')
        def index():
            return '''
            <h1>优化系统 Web 皮肤</h1>
            <form action="/optimize" method="post">
                维度: <input name="dims" value="2"><br>
                边界 (格式: -5,5 每个维度一行): <br>
                <textarea name="bounds" rows="3">-5,5
-5,5</textarea><br>
                <input type="submit" value="运行">
            </form>
            '''
        
        @app.route('/optimize', methods=['POST'])
        def optimize():
            dims = int(request.form['dims'])
            bounds_text = request.form['bounds'].strip().split('\n')
            bounds = []
            for line in bounds_text[:dims]:
                low, high = map(float, line.strip().split(','))
                bounds.append((low, high))
            
            config = OptimizationConfig(bounds, [f"x{i}" for i in range(dims)])
            orch = Orchestrator(config, source_name='source.test_function')
            best, val = orch.run('algorithm.genetic')
            return jsonify({
                'best': best.tolist(),
                'value': val,
                'status': 'success'
            })
        
        print(f"🌐 Web 服务已启动: http://{host}:{port}")
        app.run(host=host, port=port)